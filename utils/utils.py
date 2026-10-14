import csv
from training.environment import MODEL_VERSION, FRAME_COUNT, ACTION_NAMES
from stable_baselines3.common.callbacks import BaseCallback


def validate_model(model):
    """Reject old speed semantics/frame layouts before connecting hardware."""
    if (getattr(model,"model_version",None) != MODEL_VERSION
            or model.observation_space.shape not in ((FRAME_COUNT,32,32),(32,32,FRAME_COUNT))
            or getattr(model.action_space,"n",None) != len(ACTION_NAMES)):
        raise ValueError("Retrain a model with the current 32x32 binary line-mask setup")


def saved_speed(model):
    """Use the speed inside the zip, never today's config or its filename."""
    return float(model.training_config["MAX_SPEED"])


def saved_steering_gain(model):
    """Older binary checkpoints used .6; keep their action meanings intact."""
    return float(model.training_config.get("STEERING_GAIN", .6))


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
            "lane_center_error",
            "heading_error",
            "missing_line_steps",
            "time_limit",
        ])

        self.batch_writer = csv.writer(batch_file)
        self.batch_writer.writerow([
            "batch", "steps", "reward_total", "both_lines_fraction",
            "mean_abs_center_error", "missing_frames",
        ])

        self.episode = 0
        self.episode_steps = 0
        self.episode_reward = 0.0

        self.batch = 0
        self.batch_steps = 0
        self.batch_reward = 0.0
        self.batch_both_lines = 0
        self.batch_center_error = 0.0
        self.batch_center_samples = 0
        self.batch_missing_frames = 0

    def _on_step(self):
        reward = float(self.locals["rewards"][0])

        self.episode_steps += 1
        self.episode_reward += reward

        self.batch_steps += 1
        self.batch_reward += reward
        info = self.locals["infos"][0]
        if info.get("left_line_type") is not None and info.get("right_line_type") is not None:
            self.batch_both_lines += 1
        center_error = info.get("lane_center_error")
        if center_error is not None:
            self.batch_center_error += abs(float(center_error))
            self.batch_center_samples += 1
        if info.get("left_line_type") is None and info.get("right_line_type") is None:
            self.batch_missing_frames += 1
        # DQN collects only four steps before an update. Log useful groups
        # of 256 steps instead of writing a CSV row for every small update.
        if self.batch_steps == 256:
            self._write_batch()

        if self.locals["dones"][0]:
            self.episode += 1
            self.episode_writer.writerow([
                self.episode,
                self.episode_steps,
                self.episode_reward,
                info.get("termination_reason"),
                info.get("left_line_type"),
                info.get("right_line_type"),
                info.get("line_alignment"),
                info.get("lane_center_error"),
                info.get("heading_error"),
                info.get("missing_line_steps"),
                info.get("TimeLimit.truncated", False),
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
            self.batch_both_lines / max(self.batch_steps, 1),
            self.batch_center_error / max(self.batch_center_samples, 1),
            self.batch_missing_frames,
        ])
        self.batch_file.flush()
        self.batch_steps = 0
        self.batch_reward = 0.0
        self.batch_both_lines = 0
        self.batch_center_error = 0.0
        self.batch_center_samples = 0
        self.batch_missing_frames = 0

    def _on_training_end(self):
        if self.batch_steps:
            self._write_batch()
