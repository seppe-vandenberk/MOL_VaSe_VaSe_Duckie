import csv
from training.environment import MODEL_VERSION
from stable_baselines3.common.callbacks import BaseCallback


def validate_model(model):
    """Reject old speed semantics/frame layouts before connecting hardware."""
    if (getattr(model,"model_version",None) != MODEL_VERSION
            or model.observation_space.shape not in ((9,84,84),(84,84,9))
            or getattr(model.action_space,"n",None) != 5):
        raise ValueError("Use a model trained with the current three-frame RGB setup")


def saved_speed(model):
    """Use the speed inside the zip, never today's config or its filename."""
    return float(model.training_config["MAX_SPEED"])


class SaveRewards(BaseCallback):
    def __init__(self, episode_file, batch_file):
        super().__init__()

        self.episode_file = episode_file
        self.batch_file = batch_file
        self.episode_writer = csv.writer(episode_file)
        self.episode_writer.writerow([
            "episode",
            "steps",
            "reward_total",
            "termination_reason",
            "left_line_type",
            "right_line_type",
            "line_alignment",
            "missing_line_steps",
        ])

        self.batch_writer = csv.writer(batch_file)
        self.batch_writer.writerow(["batch", "steps", "reward_total"])

        self.episode = 0
        self.episode_steps = 0
        self.episode_reward = 0.0

        self.batch = 0
        self.batch_steps = 0
        self.batch_reward = 0.0

    def _on_step(self):
        reward = float(self.locals["rewards"][0])

        self.episode_steps += 1
        self.episode_reward += reward

        self.batch_steps += 1
        self.batch_reward += reward
        # DQN collects only four steps before an update. Log useful groups
        # of 256 steps instead of writing a CSV row for every small update.
        if self.batch_steps == 256:
            self._write_batch()

        if self.locals["dones"][0]:
            self.episode += 1
            info = self.locals["infos"][0]

            self.episode_writer.writerow([
                self.episode,
                self.episode_steps,
                self.episode_reward,
                info.get("termination_reason"),
                info.get("left_line_type"),
                info.get("right_line_type"),
                info.get("line_alignment"),
                info.get("missing_line_steps"),
            ])
            self.episode_file.flush()

            self.episode_steps = 0
            self.episode_reward = 0.0

        return True

    def _write_batch(self):
        self.batch += 1
        self.batch_writer.writerow([
            self.batch,
            self.batch_steps,
            self.batch_reward,
        ])
        self.batch_file.flush()
        self.batch_steps = 0
        self.batch_reward = 0.0

    def _on_training_end(self):
        if self.batch_steps:
            self._write_batch()
