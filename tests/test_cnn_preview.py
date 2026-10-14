"""Offline checks that the preview displays the actual policy frame history."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import numpy as np

from scripts.visualizer import CameraPreview
from training.network import DrivingCNN
from training.environment import CAMERA_SETTINGS, MODEL_VERSION, FRAME_COUNT, ACTION_NAMES, FrameStackCamera, RoadCamera, DiscreteSteering, line_mask


class FakeCamera(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (480, 640, 3), np.uint8)
    action_space = gym.spaces.Discrete(1)
    camera_settings = CAMERA_SETTINGS

    def reset(self, **kwargs):
        self.value = 255
        return np.full(self.observation_space.shape, self.value, np.uint8), {}

    def step(self, action):
        self.value = (self.value + 10) % 256
        return np.full(self.observation_space.shape, self.value, np.uint8), 0, False, False, {}


class CNNPreviewTests(unittest.TestCase):
    def test_display_matches_stack_and_keeps_overlays_out_of_cnn(self):
        env = FrameStackCamera(RoadCamera(FakeCamera(), preview=True))
        viewer = env.env.viewer
        # Keep the real figure creation and rendering, using the headless backend.
        with patch.object(plt, "switch_backend"), patch("scripts.visualizer.time.monotonic", side_effect=[1, 2, 3, 4]):
            try:
                observation, _ = env.reset()
                for _ in range(2):
                    observation, *_ = env.step(0)
                self.assertEqual(observation[0, 0].tolist(), [0] * FRAME_COUNT)
                for index, plot in enumerate(viewer.cnn_plots):
                    np.testing.assert_array_equal(plot.get_array(), observation[:, :, index])
                self.assertIn(f"{FRAME_COUNT}×32×32", viewer.figure._suptitle.get_text())
                status = SimpleNamespace(_selected_mask=np.ones((32, 32), np.uint8))
                viewer.show(env.env.cropped_image, status, cnn_input=observation)
                self.assertFalse(np.array_equal(viewer.detection_plot.get_array(), env.env.cropped_image))
                np.testing.assert_array_equal(viewer.cnn_plots[-1].get_array(), observation[:, :, -1])
            finally:
                env.close()

    def test_reset_preview_repeats_new_initial_frame(self):
        env = FrameStackCamera(RoadCamera(FakeCamera(), preview=True))
        with patch.object(CameraPreview, "show") as show:
            try:
                env.reset()
                env.step(0)
                observation, _ = env.reset()
                np.testing.assert_array_equal(show.call_args.kwargs["cnn_input"], observation)
                self.assertEqual(observation[0, 0].tolist(), [255] * FRAME_COUNT)
            finally:
                env.close()

    def test_paint_colors_become_white_and_background_black(self):
        image = np.array([[(255,255,255), (255,255,0), (0,0,0),
                           (60,60,60), (255,0,0), (0,0,255), (0,255,0)]], np.uint8)
        mask = line_mask(image)
        self.assertEqual(mask.shape, (1, 7, 1))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask[0, :, 0].tolist(), [255,255,0,0,0,0,0])

    def test_policy_receives_binary_channels(self):
        import torch
        from stable_baselines3 import DQN
        from stable_baselines3.common.preprocessing import preprocess_obs
        from utils.utils import validate_model
        torch.set_num_threads(1)
        env = DiscreteSteering(FrameStackCamera(RoadCamera(FakeCamera())))
        try:
            model = DQN("CnnPolicy", env, device="cpu", buffer_size=16,
                        policy_kwargs={"features_extractor_class":DrivingCNN,"net_arch":[]})
            model.model_version = MODEL_VERSION
            validate_model(model)
            observation, _ = env.reset()
            tensor, _ = model.policy.obs_to_tensor(observation)
            self.assertEqual(tuple(tensor.shape), (1, FRAME_COUNT, 32, 32))
            normalized = preprocess_obs(tensor, model.observation_space)
            self.assertEqual(torch.unique(normalized).tolist(), [1.0])
            self.assertEqual(tuple(model.q_net(tensor).shape), (1, len(ACTION_NAMES)))
            model.model_version = "simple_rgb84_stack3_v1"
            with self.assertRaisesRegex(ValueError, "Retrain"):
                validate_model(model)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
