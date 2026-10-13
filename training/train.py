"""Basic CNN-DQN: three cropped RGB frames -> five steering choices."""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0,str(PROJECT_ROOT))

import os
from datetime import datetime
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from training.environment import DiscreteSteering, FrameStackCamera, RoadCamera, MODEL_VERSION, ACTION_NAMES
from training.reward import LineReward
from utils.utils import SaveRewards

TOTAL_TIMESTEPS = 50_000


def main():
    with open(PROJECT_ROOT / "config.yaml") as stream:
        config = yaml.safe_load(stream)
    speed = float(config["MAX_SPEED"])
    learning_rate = float(config["LEARNING_RATE"])
    models = PROJECT_ROOT / "models"
    logs = PROJECT_ROOT / "training" / "logs"
    models.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
    prefix = f"dqn_simple_{timestamp}_lr{learning_rate:g}_speed{speed:g}"
    env = DiscreteSteering(
        FrameStackCamera(RoadCamera(LineReward(),preview=bool(os.environ.get("DISPLAY")))),
        speed=speed,
    )
    try:
        # These are the settings recovered from the original RGB checkpoint.
        model = DQN(
            "CnnPolicy",env,device="cuda",learning_rate=learning_rate,
            buffer_size=10000,learning_starts=1000,batch_size=64,
            gamma=.99,train_freq=4,gradient_steps=1,target_update_interval=10000,
            exploration_initial_eps=1.0,exploration_final_eps=.05,exploration_fraction=.50,
            verbose=1,
        )
        print(f"Learning device: {model.device}; {TOTAL_TIMESTEPS:,} steps; 3 RGB frames; 5 actions.")
        model.model_version = MODEL_VERSION
        model.observation_color_space = "RGB"
        model.observation_projection = "camera_crop_v1"
        model.camera_settings = dict(env.unwrapped.camera_settings)
        model.action_names = list(ACTION_NAMES)
        model.training_config = {"MAX_SPEED":speed,"LEARNING_RATE":learning_rate,"TOTAL_TIMESTEPS":TOTAL_TIMESTEPS}
        model.reward_settings = env.unwrapped.reward_settings()
        try:
            with open(logs / "episode_rewards.csv","w",newline="") as episodes, \
                 open(logs / "batch_rewards.csv","w",newline="") as batches:
                model.learn(total_timesteps=TOTAL_TIMESTEPS,callback=[
                    SaveRewards(episodes,batches),
                    CheckpointCallback(save_freq=5000,save_path=str(models),name_prefix=prefix),
                ])
        except KeyboardInterrupt:
            print("Interrupted; saving.")
        finally:
            path = models / f"{prefix}_final_{model.num_timesteps}_steps.zip"
            model.save(path)
            print(f"Saved: {path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
