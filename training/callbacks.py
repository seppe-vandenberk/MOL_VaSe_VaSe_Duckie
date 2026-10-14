"""Wall-clock budget and diagnostics for short simulator training runs."""

import csv
import time
from collections import Counter

from stable_baselines3.common.callbacks import BaseCallback

from training.environment import ACTION_NAMES


class TrainingProgress(BaseCallback):
    def __init__(self, stream, deadline, interval=256):
        super().__init__()
        self.deadline = deadline
        self.interval = interval
        self.writer = csv.DictWriter(stream, fieldnames=[
            "steps", "seconds", "fps", "reward_per_step", "failures", "time_limits",
            "both_lines_fraction", "one_line_fraction", "mean_abs_center_error",
            "epsilon", "updates", "loss", *ACTION_NAMES,
        ])
        self.writer.writeheader()
        self.stream = stream
        self.stop_reason = "step limit"
        self.started = time.monotonic()
        self._clear()

    def _clear(self):
        self.count = self.reward = self.failures = self.time_limits = 0
        self.detections = Counter()
        self.actions = Counter()
        self.errors = []

    def _on_training_start(self):
        self.started = time.monotonic()

    def _on_step(self):
        self.count += 1
        self.reward += float(self.locals["rewards"][0])
        info = self.locals["infos"][0]
        self.detections[info.get("detection_state")] += 1
        self.actions[int(self.locals["actions"][0])] += 1
        if info.get("lane_center_error") is not None:
            self.errors.append(abs(info["lane_center_error"]))
        if self.locals["dones"][0]:
            if info.get("TimeLimit.truncated", False):
                self.time_limits += 1
            else:
                self.failures += 1
        expired = time.monotonic() >= self.deadline
        if self.count >= self.interval or expired:
            self._write()
        if expired:
            self.stop_reason = "wall-clock limit"
        return not expired

    def _write(self):
        if not self.count:
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        row = dict(
            steps=self.num_timesteps, seconds=elapsed, fps=self.num_timesteps / elapsed,
            reward_per_step=self.reward / self.count, failures=self.failures,
            time_limits=self.time_limits,
            both_lines_fraction=self.detections["both"] / self.count,
            one_line_fraction=self.detections["one"] / self.count,
            mean_abs_center_error=sum(self.errors) / len(self.errors) if self.errors else "",
            epsilon=self.model.exploration_rate, updates=self.model._n_updates,
            loss=self.model.logger.name_to_value.get("train/loss", ""),
        )
        row.update({name: self.actions[i] for i, name in enumerate(ACTION_NAMES)})
        self.writer.writerow(row)
        self.stream.flush()
        print(f"steps={row['steps']} fps={row['fps']:.1f} reward/step={row['reward_per_step']:.3f} "
              f"failures={self.failures} both_lines={row['both_lines_fraction']:.0%} "
              f"epsilon={row['epsilon']:.3f} updates={row['updates']}", flush=True)
        self._clear()

    def _on_training_end(self):
        self._write()
