"""Raw RGB camera, three-frame history, and five forward steering actions."""

from collections import deque
import cv2
import gymnasium as gym
import numpy as np
from scripts.visualizer import CameraPreview

CAMERA_SETTINGS = {"crop_top": .35, "color_space": "RGB"}
FRAME_COUNT = 3
MODEL_VERSION = "simple_rgb84_stack3_v1"
ACTION_NAMES = ("hard_right", "slight_right", "straight", "slight_left", "hard_left")


def camera_input(image, settings=None):
    settings = CAMERA_SETTINGS if settings is None else settings
    top = int(image.shape[0] * settings["crop_top"])
    return cv2.resize(image[top:], (84,84), interpolation=cv2.INTER_AREA)


def steering_wheels(action, speed):
    """Share a total PWM budget; the hardest turn has a 4:1 wheel ratio."""
    if not 0 < speed <= 1:
        raise ValueError("Speed must be between 0 and 1")
    turn = .6 * DiscreteSteering.steering_values[int(action)]
    return np.array([speed * .5 * (1 - turn), speed * .5 * (1 + turn)], np.float32)


class CameraHistory:
    def __init__(self):
        self.frames = deque(maxlen=FRAME_COUNT)

    def push(self, image):
        if not self.frames:
            self.frames.extend(image.copy() for _ in range(FRAME_COUNT))
        else:
            self.frames.append(image.copy())
        return np.concatenate(self.frames, axis=2)


class RoadCamera(gym.ObservationWrapper):
    def __init__(self, env, preview=False):
        super().__init__(env)
        self.viewer = CameraPreview() if preview else None
        self.observation_space = gym.spaces.Box(0,255,(84,84,3),np.uint8)

    def observation(self, image):
        cropped = camera_input(image, self.unwrapped.camera_settings)
        if self.viewer:
            self.viewer.show(cropped, self.unwrapped, raw_image=image)
        return cropped

    def close(self):
        try:
            self.env.close()
        finally:
            if self.viewer:
                self.viewer.close()


class FrameStackCamera(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.history = CameraHistory()
        self.observation_space = gym.spaces.Box(0,255,(84,84,3 * FRAME_COUNT),np.uint8)

    def reset(self, **kwargs):
        image, info = self.env.reset(**kwargs)
        self.history = CameraHistory()
        return self.history.push(image), info

    def step(self, action):
        image, reward, terminated, truncated, info = self.env.step(action)
        return self.history.push(image), reward, terminated, truncated, info


class DiscreteSteering(gym.ActionWrapper):
    steering_values = (-1., -.5, 0., .5, 1.)

    def __init__(self, env, speed=.45):
        super().__init__(env)
        self.speed = speed
        self.action_space = gym.spaces.Discrete(5)

    def action(self, action):
        return steering_wheels(action, self.speed)

