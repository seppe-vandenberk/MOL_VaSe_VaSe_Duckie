"""Lane-centering reward computed from the same RGB crop seen by DQN."""

import time
import gymnasium as gym
import numpy as np
from duckietown.sdk.robots.duckiebot import DB21J
from training.environment import CAMERA_SETTINGS, camera_input
from utils.line_detection import LineDetector


class LineReward(LineDetector, gym.Env):
    # Keep rewards near unit scale so Q targets do not grow into the thousands.
    one_line_penalty = .5
    no_lines_penalty = 1.
    max_missing_line_steps = 5
    lost_lines_failure_penalty = 4.
    heading_penalty = .25
    observation_interval = .05
    max_partial_line_steps = 40
    max_episode_steps = 1200
    reset_settle_seconds = .15
    reward_version = "lane_recovery_color_v3"

    def __init__(self, entity_name="map_0/vehicle_0", camera_settings=None, port=7501):
        super().__init__()
        self.camera_settings = dict(CAMERA_SETTINGS if camera_settings is None else camera_settings)
        self.robot = DB21J(entity_name, simulated=True, port=port)
        self.components = (self.robot.camera, self.robot.motors, self.robot.reset_flag)
        try:
            for component in self.components:
                component.start()
        except BaseException:
            self.close()
            raise
        self.action_space = gym.spaces.Box(-1.,1.,(2,),np.float32)
        self.observation_space = gym.spaces.Box(0,255,(480,640,3),np.uint8)
        self.last_frame_time = None

    def reward_settings(self):
        return {name: getattr(self,name) for name in (
            "one_line_penalty", "no_lines_penalty", "max_missing_line_steps",
            "lost_lines_failure_penalty", "heading_penalty", "max_partial_line_steps",
            "max_episode_steps")}

    def _camera(self):
        if self.last_frame_time is not None:
            delay = self.observation_interval - (time.monotonic() - self.last_frame_time)
            if delay > 0:
                time.sleep(delay)
        # Measure the period from the capture request, so waiting for a camera
        # frame does not add another full period to every step.
        self.last_frame_time = time.monotonic()
        frame = self.robot.camera.capture(block=True, timeout=2)
        if frame is None:
            raise RuntimeError("Simulator camera timed out; check that Duckiematrix is running and unpaused")
        image = frame[:, :, [2,1,0]]
        self._reward_image = camera_input(image,self.camera_settings)
        return image

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.robot.motors.set_pwm(left=0.,right=0.)
        self.robot.reset_flag.set_reset(True)
        if self.reset_settle_seconds:
            time.sleep(self.reset_settle_seconds)
        self.last_frame_time = None
        self.missing_steps = 0
        self.partial_steps = 0
        self.episode_steps = 0
        self._lane_width = None
        self._top_lane_width = None
        self._lane_center_error = None
        self._heading_error = None
        self._detection_state = "none"
        self.reason = None
        self._line_alignment = 0.
        image = self._camera()
        # Establish geometry for recovery immediately, including on reset.
        self.reward_fn()
        self.missing_steps = self.partial_steps = 0
        self.reason = None
        return image, self._get_info()

    def _alignment_reward(self, bottom_center, top_center, half_lane_width):
        width = self._reward_image.shape[1]
        self._lane_center_error = float((bottom_center - (width - 1) / 2) / half_lane_width)
        self._heading_error = float((top_center - bottom_center) / width)
        self._line_alignment = 1. - min(abs(self._lane_center_error), 1.)
        heading = min(abs(self._heading_error) / .25, 1.)
        return self._line_alignment - self.heading_penalty * heading

    def reward_fn(self):
        left, right = self.detect(self._reward_image)
        self._line_alignment = 0.
        self._lane_center_error = None
        self._heading_error = None
        self.reason = None
        width = self._reward_image.shape[1]
        if left is not None and right is not None and right[1] - left[1] >= width * .2:
            self.missing_steps = 0
            self.partial_steps = 0
            self._detection_state = "both"
            bottom_center = (left[2][0] + right[2][0]) / 2
            top_center = (left[2][2] + right[2][2]) / 2
            half_lane_width = max((right[1] - left[1]) / 2, width * .1)
            self._lane_width = float(right[2][0] - left[2][0])
            self._top_lane_width = float(right[2][2] - left[2][2])
            reward = self._alignment_reward(bottom_center, top_center, half_lane_width)
            return float(reward), False
        self.partial_steps = getattr(self, "partial_steps", 0) + 1
        if (left is None) != (right is None):
            self.missing_steps = 0
            self._detection_state = "one"
            reward = -self.one_line_penalty
            # A recent, valid pair supplies lane width. Never invent geometry
            # when an episode starts with only one boundary visible.
            lane_width = getattr(self, "_lane_width", None)
            if lane_width is not None and self.partial_steps <= self.max_partial_line_steps:
                line = left if left is not None else right
                direction = 1 if left is not None else -1
                bottom = line[2][0] + direction * lane_width / 2
                top = line[2][2] + direction * self._top_lane_width / 2
                alignment = self._alignment_reward(bottom, top, max(lane_width / 2, width * .1))
                reward -= .5 * (1. - alignment)
        else:
            self._detection_state = "none"
            self.missing_steps += 1
            reward = -self.no_lines_penalty
        if self.missing_steps >= self.max_missing_line_steps:
            self.reason = f"both lines missing for {self.missing_steps} steps"
        elif self.partial_steps >= self.max_partial_line_steps:
            self.reason = f"lane not recovered for {self.partial_steps} steps"
        if self.reason:
            return float(reward - self.lost_lines_failure_penalty), True
        return float(reward), False

    def step(self, action):
        left, right = action
        self.robot.motors.set_pwm(left=float(left),right=float(right))
        image = self._camera()
        reward, terminated = self.reward_fn()
        self.episode_steps += 1
        truncated = not terminated and self.episode_steps >= self.max_episode_steps
        if truncated:
            self.reason = "episode time limit"
        if terminated or truncated:
            self.robot.motors.set_pwm(left=0.,right=0.)
        return image, reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {"left_line_type": self._left_line_type, "right_line_type": self._right_line_type,
                "line_alignment": self._line_alignment, "missing_line_steps": self.missing_steps,
                "lane_center_error": getattr(self, "_lane_center_error", None),
                "heading_error": getattr(self, "_heading_error", None),
                "detection_state": self._detection_state,
                "partial_line_steps": self.partial_steps,
                "termination_reason": self.reason}

    def close(self):
        try:
            if self.robot.motors.is_started:
                self.robot.motors.set_pwm(left=0.,right=0.)
        finally:
            for component in self.components:
                if component.is_started:
                    component.stop()
