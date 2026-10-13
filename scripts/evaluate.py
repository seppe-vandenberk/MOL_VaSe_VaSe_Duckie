"""Deterministically evaluate a saved DQN line-following policy."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time

from stable_baselines3 import DQN

from training.environment import DiscreteSteering, RoadCamera, FrameStackCamera
from training.reward import LineReward
from utils.utils import saved_speed, validate_model


def evaluate_run(model, env, duration):
    """Drive for a fixed time, resetting after failures within that time."""
    observation, _ = env.reset()
    started = time.monotonic()
    total_reward = 0.0
    steps = 0
    failures = 0
    while time.monotonic() - started < duration:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            failures += 1
            print(f"  reset: {info.get('termination_reason')}")
            if time.monotonic() - started < duration:
                observation, _ = env.reset()
    env.unwrapped.robot.motors.set_pwm(left=0.0, right=0.0)
    elapsed = time.monotonic() - started
    print(
        f"seconds={elapsed:.1f} steps={steps} failures={failures} "
        f"reward_total={total_reward:.3f} "
        f"reward_per_step={total_reward / max(steps, 1):.3f}"
    )
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="saved zip; defaults to newest in models/")
    parser.add_argument("--runs", "--episodes", dest="runs", type=int, default=5)
    parser.add_argument("--duration", type=float, default=60.0,
                        help="seconds per run (default: 60)")
    parser.add_argument("--entity", default="map_0/vehicle_0",
                        help="use a different vehicle when training is running")
    arguments = parser.parse_args()

    if arguments.model:
        model_path = Path(arguments.model)
    else:
        saved_models = list((PROJECT_ROOT / "models").glob("dqn_simple_*.zip"))
        model_path = (
            max(saved_models, key=lambda path: path.stat().st_mtime_ns)
            if saved_models else PROJECT_ROOT / "models" / "no_camera_rgb_model.zip"
        )

    print(f"Evaluating model: {model_path}")
    model = DQN.load(model_path, device="auto")
    try:
        speed = saved_speed(model)
        validate_model(model)
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved training configuration: {model.training_config}")
    reward_env = LineReward(
            entity_name=arguments.entity,
            camera_settings=model.camera_settings,
        )
    for name, value in getattr(model, "reward_settings", {}).items():
        if name in reward_env.reward_settings():
            setattr(reward_env, name, value)
    env = DiscreteSteering(
        FrameStackCamera(RoadCamera(reward_env)), speed=speed,
    )
    returns = []
    try:
        for run in range(1, arguments.runs + 1):
            print(f"run={run}/{arguments.runs}")
            returns.append(evaluate_run(model, env, arguments.duration))
    finally:
        env.close()

    print(
        f"mean_return={sum(returns) / len(returns):.3f} "
        f"runs={len(returns)}"
    )


if __name__ == "__main__":
    main()
