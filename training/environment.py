"""Paper-inspired segmented camera input and five forward steering actions."""

import gymnasium as gym
import cv2
import numpy as np
from collections import deque
from scripts.visualizer import CameraPreview


CAMERA_SETTINGS = {"resize_width": 80, "resize_height": 60, "crop_rows": 20,
                   "color_space": "SEGMENTED_RGB", "yellow_low": [12,60,80],
                   "yellow_high": [42,255,255], "white_low": [0,0,120],
                   "white_high": [179,100,255]}
FRAME_COUNT = 5
MODEL_VERSION = "paper5_segmented_40x80_v1"
ACTION_NAMES = ("hard_right", "slight_right", "straight", "slight_left", "hard_left")


def camera_input(image, settings=None):
    """Resize to 60x80, crop to 40x80; yellow -> red, white -> green."""
    settings = CAMERA_SETTINGS if settings is None else settings
    resized = cv2.resize(image, (settings["resize_width"], settings["resize_height"]), interpolation=cv2.INTER_AREA)
    cropped = resized[settings["crop_rows"]:]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array(settings["yellow_low"]), np.array(settings["yellow_high"]))
    white = cv2.inRange(hsv, np.array(settings["white_low"]), np.array(settings["white_high"]))
    return np.stack((yellow, white, np.zeros_like(yellow)), axis=2)


def steering_wheels(action, speed):
    """Distribute a total speed budget across forward wheels, up to 4:1."""
    index = int(np.asarray(action).item())
    if not 0 <= index < len(DiscreteSteering.steering_values):
        raise ValueError(f"invalid steering action: {action}")
    if not np.isfinite(speed) or not 0.05 <= speed <= 1.0:
        raise ValueError("total speed must be between 0.05 and 1.0")
    # (1 + 0.6) / (1 - 0.6) = 4; left + right stays equal to speed.
    turn = 0.6 * float(DiscreteSteering.steering_values[index])
    return np.array([speed * 0.5 * (1.0 - turn), speed * 0.5 * (1.0 + turn)], dtype=np.float32)


class CameraHistory:
    """Five RGB frames in chronological order; repeat the first at startup."""

    def __init__(self):
        self.frames = deque(maxlen=FRAME_COUNT)

    def push(self, image):
        if not self.frames:
            self.frames.extend(image.copy() for _ in range(FRAME_COUNT))
        else:
            self.frames.append(image.copy())
        return np.concatenate(self.frames, axis=2)


class FrameStackCamera(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.history = CameraHistory()
        self.observation_space = gym.spaces.Box(0, 255, (40, 80, 3 * FRAME_COUNT), np.uint8)

    def reset(self, **kwargs):
        image, info = self.env.reset(**kwargs)
        self.history = CameraHistory()
        return self.history.push(image), info

    def step(self, action):
        image, reward, terminated, truncated, info = self.env.step(action)
        return self.history.push(image), reward, terminated, truncated, info


class RoadCamera(gym.ObservationWrapper):
    """Provide identical segmentation to training, evaluation, and the preview."""

    def __init__(self, env, preview=False):
        super().__init__(env)
        self.viewer = CameraPreview() if preview else None
        self.observation_space = gym.spaces.Box(0, 255, (40, 80, 3), np.uint8)

    def observation(self, image):
        observation = camera_input(image, self.unwrapped.camera_settings)
        if self.viewer is not None:
            self.viewer.show(observation, self.unwrapped, raw_image=image)
        return observation

    def close(self):
        try:
            self.env.close()
        finally:
            if self.viewer is not None:
                self.viewer.close()


class DiscreteSteering(gym.ActionWrapper):
    steering_values = np.array([-1., -.5, 0., .5, 1.], dtype=np.float32)

    def __init__(self, env, speed=0.45, pwm_scale=1.0):
        super().__init__(env)
        self.speed = speed
        self.pwm_scale = pwm_scale
        self.action_space = gym.spaces.Discrete(len(self.steering_values))

    def action(self, action):
        return steering_wheels(action, self.speed) * self.pwm_scale

