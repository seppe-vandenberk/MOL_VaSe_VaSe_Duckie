"""Binary white/yellow road paint, camera history, and steering actions."""

from collections import deque
import cv2
import gymnasium as gym
import numpy as np
from scripts.visualizer import CameraPreview, AsyncCameraPreview
from utils.paint_mask import paint_masks

CAMERA_SETTINGS = {
    "crop_top": .50, "color_space": "RGB",
    "white_hsv_lower": [0, 0, 120], "white_hsv_upper": [179, 100, 255],
    "yellow_hsv_lower": [12, 60, 80], "yellow_hsv_upper": [42, 255, 255],
}
FRAME_COUNT = 1
MODEL_VERSION = "binary_paint32x32_v1"
ACTION_NAMES = ("right", "slight_right", "straight", "slight_left", "left")


def camera_input(image, settings=None):
    settings = CAMERA_SETTINGS if settings is None else settings
    top = int(image.shape[0] * settings["crop_top"])
    return cv2.resize(image[top:], (32, 32), interpolation=cv2.INTER_AREA)


def line_mask(image, settings=None):
    """Convert a cropped RGB image to one uint8 channel: paint=255, road=0."""
    settings = CAMERA_SETTINGS if settings is None else settings
    return cv2.bitwise_or(*paint_masks(image, settings))[:, :, None]


def steering_wheels(action, speed, steering_gain=.6):
    """Share total PWM; gain .6 gives a 2.125:1 ratio, gain 1 gives 4:1."""
    if not 0 < speed <= 1:
        raise ValueError("Speed must be between 0 and 1")
    if not 0 < steering_gain <= 1:
        raise ValueError("Steering gain must be between 0 and 1")
    if int(action) != action or not 0 <= int(action) < len(ACTION_NAMES):
        raise ValueError("Invalid steering action")
    turn = steering_gain * DiscreteSteering.steering_values[int(action)]
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
    def __init__(self, env, preview=False, asynchronous_preview=False):
        super().__init__(env)
        viewer_type = AsyncCameraPreview if asynchronous_preview else CameraPreview
        self.viewer = viewer_type() if preview else None
        self.raw_image = None
        self.cropped_image = None
        self.observation_space = gym.spaces.Box(0,255,(32,32,1),np.uint8)

    def observation(self, image):
        cropped = camera_input(image, self.unwrapped.camera_settings)
        self.raw_image = image
        self.cropped_image = cropped
        return line_mask(cropped, self.unwrapped.camera_settings)

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
        self.observation_space = gym.spaces.Box(0,255,(32,32,FRAME_COUNT),np.uint8)

    def reset(self, **kwargs):
        image, info = self.env.reset(**kwargs)
        self.history = CameraHistory()
        return self._stack_and_preview(image), info

    def _stack_and_preview(self, image):
        observation = self.history.push(image)
        if self.env.viewer:
            self.env.viewer.show(
                self.env.cropped_image, self.unwrapped, raw_image=self.env.raw_image,
                cnn_input=observation,
            )
        return observation

    def step(self, action):
        image, reward, terminated, truncated, info = self.env.step(action)
        return self._stack_and_preview(image), reward, terminated, truncated, info


class DiscreteSteering(gym.ActionWrapper):
    steering_values = (-0.6, -0.3, 0.0, 0.3, 0.6)

    def __init__(self, env, speed=.45, steering_gain=.6):
        super().__init__(env)
        self.speed = speed
        self.steering_gain = steering_gain
        steering_wheels(2, speed, steering_gain)
        self.action_space = gym.spaces.Discrete(5)

    def action(self, action):
        return steering_wheels(action, self.speed, self.steering_gain)

