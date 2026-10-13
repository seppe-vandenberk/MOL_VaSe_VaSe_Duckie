"""Train the paper-inspired five-action Duckiematrix DQN on CUDA."""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
from datetime import datetime
import torch
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from training.environment import DiscreteSteering, RoadCamera, FrameStackCamera, MODEL_VERSION, ACTION_NAMES
from training.network import DrivingCNN
from training.reward import LaneReward
from utils.utils import SaveRewards

TOTAL_TIMESTEPS = 24_000
LEARNING_STARTS = 1_000  # Shortened from the paper's 10,000 for this short run.
BUFFER_SIZE = 50_000
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = .05
EXPLORATION_FRACTION = .60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    with open(PROJECT_ROOT / "config.yaml") as stream:
        config = yaml.safe_load(stream)
    speed = float(config["MAX_SPEED"])
    learning_rate = float(config["LEARNING_RATE"])
    if not 0 < learning_rate <= .003:
        raise ValueError("LEARNING_RATE must be positive and at most 0.003")
    if not .05 <= speed <= 1.0:
        raise ValueError("MAX_SPEED must be between 0.05 and 1.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    torch.set_num_threads(2)
    print(f"CUDA: {torch.cuda.get_device_name(0)}; {TOTAL_TIMESTEPS:,} steps at up to 20 FPS.")
    print(f"Total wheel budget={speed:g}; straight={speed / 2:g} per wheel; learning rate={learning_rate:g}")
    models_dir = PROJECT_ROOT / "models"
    logs_dir = PROJECT_ROOT / "training" / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
    prefix = f"dqn_paper5_{timestamp}_lr{learning_rate:g}_speed{speed:g}"
    preview = not args.no_preview and bool(os.environ.get("DISPLAY"))
    env = DiscreteSteering(FrameStackCamera(RoadCamera(LaneReward(), preview=preview)), speed=speed)
    try:
        model = DQN(
            "CnnPolicy", env, device="cuda",
            policy_kwargs={"features_extractor_class": DrivingCNN, "net_arch": []},
            learning_rate=learning_rate, batch_size=32, gamma=.99,
            buffer_size=BUFFER_SIZE, learning_starts=LEARNING_STARTS,
            train_freq=4, gradient_steps=1, target_update_interval=1000,
            exploration_initial_eps=EXPLORATION_INITIAL_EPS,
            exploration_final_eps=EXPLORATION_FINAL_EPS,
            exploration_fraction=EXPLORATION_FRACTION, verbose=1,
        )
        model.model_version = MODEL_VERSION
        model.observation_color_space = "SEGMENTED_RGB"
        model.observation_projection = "paper_camera_v1"
        model.camera_settings = dict(env.unwrapped.camera_settings)
        model.action_names = list(ACTION_NAMES)
        model.speed_semantics = "total_wheel_pwm"
        model.pwm_scale = 1.
        model.reward_profile = env.unwrapped.REWARD_VERSION
        model.reward_settings = env.unwrapped.reward_settings()
        model.training_config = {
            "MAX_SPEED": speed, "LEARNING_RATE": learning_rate, "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
            "BUFFER_SIZE": BUFFER_SIZE, "LEARNING_STARTS": LEARNING_STARTS,
            "BATCH_SIZE": 32, "GAMMA": .99, "TRAIN_FREQ": 4, "GRADIENT_STEPS": 1,
            "TARGET_UPDATE_INTERVAL": 1000,
            "EXPLORATION_INITIAL_EPS": EXPLORATION_INITIAL_EPS,
            "EXPLORATION_FINAL_EPS": EXPLORATION_FINAL_EPS,
            "EXPLORATION_FRACTION": EXPLORATION_FRACTION,
        }
        try:
            with open(logs_dir / "episode_rewards.csv", "w", newline="") as episodes, \
                 open(logs_dir / "batch_rewards.csv", "w", newline="") as batches:
                model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[
                    SaveRewards(episodes, batches),
                    CheckpointCallback(save_freq=5000, save_path=str(models_dir), name_prefix=prefix),
                ])
        except KeyboardInterrupt:
            print("Interrupted; saving the current model.")
        finally:
            path = models_dir / f"{prefix}_final_{model.num_timesteps}_steps.zip"
            model.save(path)
            print(f"Saved: {path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
