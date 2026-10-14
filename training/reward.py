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
    max_missing_line_steps = 10
    lost_lines_failure_penalty = 4.
    heading_penalty = .25
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
            "one_line_penalty", "no_lines_penalty", "max_missing_line_steps",
            "lost_lines_failure_penalty", "heading_penalty")}

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
        width = self._reward_image.shape[1]
        if left is not None and right is not None:
            self.missing_steps = 0
            self.reason = None
            bottom_center = (left[2][0] + right[2][0]) / 2
            top_center = (left[2][2] + right[2][2]) / 2
            half_lane_width = max((right[1] - left[1]) / 2, width * .1)
            center_error = float((bottom_center - (width - 1) / 2) / half_lane_width)
            heading_error = float((top_center - bottom_center) / width)
            center_score = 1. - min(abs(center_error), 1.)
            heading_score = min(abs(heading_error) / .25, 1.)
            self._line_alignment = center_score
            self._lane_center_error = center_error
            self._heading_error = heading_error
            reward = center_score - self.heading_penalty * heading_score
            return float(reward), False
        self._lane_center_error = None
        self._heading_error = None
        if left is not None or right is not None:
            self.missing_steps = 0
            self.reason = None
            return -self.one_line_penalty, False
        self.missing_steps += 1
        self.reason = f"both lines missing for {self.missing_steps} steps"
        if self.missing_steps >= self.max_missing_line_steps:
            return -self.no_lines_penalty - self.lost_lines_failure_penalty, True
        return -self.no_lines_penalty, False

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
                "lane_center_error": getattr(self, "_lane_center_error", None),
                "heading_error": getattr(self, "_heading_error", None),
                "termination_reason": self.reason}

    def close(self):
        self.robot.motors.set_pwm(left=0.,right=0.)
        for component in self.components:
            if component.is_started:
                component.stop()
