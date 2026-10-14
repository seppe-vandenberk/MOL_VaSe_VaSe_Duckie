"""Train a small camera DQN within a wall-clock budget, then evaluate it."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import datetime
import json
import math
import time

import torch
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from scripts.evaluate import evaluate_runs
from training.callbacks import TrainingProgress
from training.environment import DiscreteSteering, FrameStackCamera, RoadCamera, MODEL_VERSION, ACTION_NAMES
from training.reward import LineReward
from training.network import DrivingCNN
from utils.utils import SaveRewards, saved_speed, saved_steering_gain, validate_model

TOTAL_TIMESTEPS = 30_000
DQN_SETTINGS = dict(buffer_size=30_000, learning_starts=500, batch_size=64,
                    gamma=.99, train_freq=1, gradient_steps=1, target_update_interval=500,
                    exploration_initial_eps=1., exploration_final_eps=.05, exploration_fraction=.20)


def build_model(env, learning_rate, device="auto", seed=0, warm_start=None):
    """Fresh replay and schedule; optionally initialise from compatible weights."""
    model = DQN("CnnPolicy", env, device=device, learning_rate=learning_rate, seed=seed,
                policy_kwargs={"features_extractor_class": DrivingCNN, "net_arch": []},
                verbose=0, **DQN_SETTINGS)
    if warm_start is not None:
        validate_model(warm_start)
        model.policy.load_state_dict(warm_start.policy.state_dict())
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--train-minutes", type=float, default=25.)
    parser.add_argument("--budget-minutes", type=float, default=30.)
    parser.add_argument("--eval-runs", type=int, default=5, help="0 skips evaluation for short tuning runs")
    parser.add_argument("--eval-duration", type=float, default=60.)
    parser.add_argument("--preview", action="store_true", help="show camera preview (off by default)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entity", default="map_0/vehicle_0")
    parser.add_argument("--steering-gain", type=float, help=".6 keeps old steering; 1.0 enables 4:1 turns")
    parser.add_argument("--warm-start", type=Path, help="initialise compatible weights, with a fresh replay buffer")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.eval_runs < 0:
        parser.error("steps must be positive and eval-runs cannot be negative")
    for name in ("train_minutes", "budget_minutes", "eval_duration"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    if args.train_minutes > args.budget_minutes:
        parser.error("train-minutes must not exceed budget-minutes")
    if args.steering_gain is not None and not 0 < args.steering_gain <= 1:
        parser.error("steering-gain must be between 0 and 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    train_deadline = started + args.train_minutes * 60
    torch.set_num_threads(1)
    with open(PROJECT_ROOT / "config.yaml") as stream:
        config = yaml.safe_load(stream)
    speed = float(config["MAX_SPEED"])
    learning_rate = float(config["LEARNING_RATE"])
    if not 0 < speed <= 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("MAX_SPEED must be in (0, 1] and LEARNING_RATE must be positive and finite")
    steering_gain = args.steering_gain if args.steering_gain is not None else .6
    source = None
    if args.warm_start:
        source = DQN.load(args.warm_start, device=args.device)
        validate_model(source)
        speed = saved_speed(source)
        steering_gain = saved_steering_gain(source)
        if args.steering_gain is not None and args.steering_gain != steering_gain:
            raise ValueError("Warm-start steering must match the source checkpoint")
    models = PROJECT_ROOT / "models"
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
    prefix = f"dqn_simple_{timestamp}_lr{learning_rate:g}_speed{speed:g}"
    logs = PROJECT_ROOT / "training" / "logs" / prefix
    models.mkdir(exist_ok=True)
    logs.mkdir(parents=True)
    report = dict(settings=vars(args) | {"warm_start": str(args.warm_start) if args.warm_start else None},
                  dqn_settings=DQN_SETTINGS, speed=speed, learning_rate=learning_rate,
                  steering_gain=steering_gain, evaluation=None)
    report_path = logs / "summary.json"
    model = None
    env = None
    interrupted = False
    try:
        reward_env = LineReward(entity_name=args.entity,
                                camera_settings=source.camera_settings if source is not None else None)
        env = DiscreteSteering(FrameStackCamera(RoadCamera(reward_env, preview=args.preview)),
                               speed=speed, steering_gain=steering_gain)
        model = build_model(env, learning_rate, args.device, args.seed, source)
        model.model_version = MODEL_VERSION
        model.observation_color_space = "BINARY_PAINT"
        model.observation_projection = "camera_crop_hsv_paint_v1"
        model.camera_settings = dict(env.unwrapped.camera_settings)
        model.action_names = list(ACTION_NAMES)
        model.training_config = dict(MAX_SPEED=speed, LEARNING_RATE=learning_rate,
                                     STEERING_GAIN=steering_gain, TOTAL_TIMESTEPS=args.steps,
                                     SEED=args.seed, **DQN_SETTINGS)
        model.reward_settings = env.unwrapped.reward_settings()
        model.reward_version = env.unwrapped.reward_version
        model.warm_start_checkpoint = str(args.warm_start) if args.warm_start else None
        model.warm_start_steps = source.num_timesteps if source is not None else 0
        print(f"Device={model.device}; up to {args.steps:,} training steps / {args.train_minutes:g} min; "
              f"total budget={args.budget_minutes:g} min; logs={logs}", flush=True)
        with open(logs / "episode_rewards.csv", "w", newline="") as episodes, \
             open(logs / "batch_rewards.csv", "w", newline="") as batches, \
             open(logs / "progress.csv", "w", newline="") as progress:
            budget = TrainingProgress(progress, train_deadline)
            rewards = SaveRewards(episodes, batches)
            try:
                model.learn(total_timesteps=args.steps, callback=[rewards, budget,
                    CheckpointCallback(save_freq=5000, save_path=str(models), name_prefix=prefix)])
            except KeyboardInterrupt:
                interrupted = True
                budget.stop_reason = "interrupted"
                print("Interrupted; saving and stopping.", flush=True)
            except Exception:
                budget.stop_reason = "error"
                raise
            finally:
                env.unwrapped.robot.motors.set_pwm(left=0., right=0.)
                # These are idempotent after a normal finish and flush partial
                # batches on interruption or a camera error.
                rewards.on_training_end()
                budget.on_training_end()
                report.update(steps=model.num_timesteps, updates=model._n_updates,
                              training_seconds=time.monotonic() - started, stop_reason=budget.stop_reason)
                if model.num_timesteps:
                    path = models / f"{prefix}_final_{model.num_timesteps}_steps.zip"
                    model.save(path)
                    report["model"] = str(path)
                    print(f"Saved: {path}", flush=True)
        if not interrupted and args.eval_runs and time.monotonic() < deadline:
            report["evaluation"] = evaluate_runs(model, env, args.eval_runs, args.eval_duration, deadline=deadline)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            report["total_seconds"] = time.monotonic() - started
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Run summary: {report_path}", flush=True)


if __name__ == "__main__":
    main()
