"""Simple pre-HSV visual reward, on the same RGB crop seen by DQN."""

import time
import gymnasium as gym
import numpy as np
from duckietown.sdk.robots.duckiebot import DB21J
from training.environment import CAMERA_SETTINGS, camera_input
from utils.line_detection import LineDetector


class LineReward(LineDetector, gym.Env):
    both_lines_reward = 25.
    one_line_penalty = 8.
    no_lines_penalty = 10.
    edge_alignment_gain = 2.
    step_penalty = .01
    max_missing_line_steps = 10
    lost_lines_failure_penalty = 30.
    observation_interval = .05

    def __init__(self, entity_name="map_0/vehicle_0", camera_settings=None):
        super().__init__()
        self.camera_settings = dict(CAMERA_SETTINGS if camera_settings is None else camera_settings)
        self.robot = DB21J(entity_name, simulated=True)
        self.components = (self.robot.camera, self.robot.motors, self.robot.reset_flag)
        for component in self.components:
            component.start()
        self.action_space = gym.spaces.Box(-1.,1.,(2,),np.float32)
        self.observation_space = gym.spaces.Box(0,255,(480,640,3),np.uint8)
        self.last_frame_time = None

    def reward_settings(self):
        return {name: getattr(self,name) for name in (
            "both_lines_reward", "one_line_penalty", "no_lines_penalty", "edge_alignment_gain",
            "step_penalty", "max_missing_line_steps", "lost_lines_failure_penalty")}

    def _camera(self):
        if self.last_frame_time is not None:
            delay = self.observation_interval - (time.monotonic() - self.last_frame_time)
            if delay > 0:
                time.sleep(delay)
        frame = self.robot.camera.capture(block=True, timeout=2)
        if frame is None:
            raise RuntimeError("Camera timed out")
        self.last_frame_time = time.monotonic()
        image = frame[:, :, [2,1,0]]
        self._reward_image = camera_input(image,self.camera_settings)
        return image

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.robot.motors.set_pwm(left=0.,right=0.)
        self.robot.reset_flag.set_reset(True)
        self.missing_steps = 0
        self.reason = None
        self._line_alignment = 0.
        image = self._camera()
        self.detect(self._reward_image)
        return image, self._get_info()

    def reward_fn(self):
        left, right = self.detect(self._reward_image)
        self._line_alignment = 0.
        if left is not None and right is not None and left[4] == "dashed" and right[4] == "solid":
            width = self._reward_image.shape[1]
            left_score = 1 - abs(left[1]) / (width * self.edge_band_fraction)
            right_score = 1 - abs(width - 1 - right[1]) / (width * self.edge_band_fraction)
            self._line_alignment = float(np.clip((left_score + right_score) / 2,0,1))
            self.missing_steps = 0
            self.reason = None
            return self.both_lines_reward + self.edge_alignment_gain * self._line_alignment - self.step_penalty, False
        if left is not None or right is not None:
            self.missing_steps = 0
            self.reason = None
            return -self.one_line_penalty - self.step_penalty, False
        self.missing_steps += 1
        self.reason = f"both lines missing for {self.missing_steps} steps"
        if self.missing_steps >= self.max_missing_line_steps:
            return -self.no_lines_penalty - self.lost_lines_failure_penalty, True
        return -self.no_lines_penalty - self.step_penalty, False

    def step(self, action):
        left, right = action
        self.robot.motors.set_pwm(left=float(left),right=float(right))
        image = self._camera()
        reward, terminated = self.reward_fn()
        if terminated:
            self.robot.motors.set_pwm(left=0.,right=0.)
        return image, reward, terminated, False, self._get_info()

    def _get_info(self):
        return {"left_line_type": self._left_line_type, "right_line_type": self._right_line_type,
                "line_alignment": self._line_alignment, "missing_line_steps": self.missing_steps,
                "termination_reason": self.reason}

    def close(self):
        self.robot.motors.set_pwm(left=0.,right=0.)
        for component in self.components:
            if component.is_started:
                component.stop()
