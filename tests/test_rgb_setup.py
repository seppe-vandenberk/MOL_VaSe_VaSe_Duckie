"""Offline checks for the 32x32 binary paint CNN-DQN."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import torch
from stable_baselines3 import DQN
from training.environment import CAMERA_SETTINGS, MODEL_VERSION, FRAME_COUNT, ACTION_NAMES, CameraHistory, DiscreteSteering, FrameStackCamera, RoadCamera, camera_input, steering_wheels
from training.reward import LineReward
from training.network import DrivingCNN
from scripts.run_real import real_wheels
from scripts.visualizer import CameraPreview
from utils.utils import validate_model


class SimpleDQNTests(unittest.TestCase):
    def test_exact_crop_and_rgb_channels(self):
        image = np.random.default_rng(0).integers(0,256,(480,640,3),dtype=np.uint8)
        expected = cv2.resize(image[240:],(32,32),interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(camera_input(image),expected)

    def test_frame_order_and_initialization(self):
        history = CameraHistory()
        for value in range(4):
            observation = history.push(np.full((32,32,1),value,np.uint8))
        self.assertEqual(observation.shape,(32,32,FRAME_COUNT))
        self.assertEqual(observation[0,0].tolist(),list(range(4 - FRAME_COUNT, 4)))
        self.assertTrue(np.all(CameraHistory().push(np.full((32,32,1),9,np.uint8)) == 9))

    def test_forward_speed_budget_and_gain(self):
        for action in range(len(ACTION_NAMES)):
            wheels = steering_wheels(action,.45)
            self.assertTrue(np.all(wheels > 0))
            self.assertAlmostEqual(float(wheels.sum()),.45,places=6)
        np.testing.assert_allclose(steering_wheels(2,.45),[.225,.225])
        np.testing.assert_allclose(steering_wheels(0,.45),[.306,.144])
        np.testing.assert_allclose(real_wheels(2,.45,2),[.45,.45])
        self.assertLessEqual(float(real_wheels(0,.45,10).max()),1)

    def test_recovered_reward_and_missing_frame_limit(self):
        env = LineReward.__new__(LineReward)
        env._reward_image = np.zeros((32,32,3),np.uint8)
        env.missing_steps = 0
        left = (1,3,(3,31,10,0),[],"dashed")
        right = (1,28,(28,31,21,0),[],"solid")
        env.detect = lambda image: (left,right)
        self.assertAlmostEqual(env.reward_fn()[0],1.)
        env.detect = lambda image: (left,None)
        self.assertAlmostEqual(env.reward_fn()[0],-.5)
        env.detect = lambda image: (None,None)
        for step in range(1,env.max_missing_line_steps + 1):
            value,ended = env.reward_fn()
            self.assertEqual(ended,step == env.max_missing_line_steps)
            self.assertAlmostEqual(value,-5. if ended else -1.)
        env.detect = lambda image: (left,right)
        self.assertAlmostEqual(env.reward_fn()[0],1.)
        self.assertEqual(env.missing_steps,0)
        self.assertEqual(env.partial_steps,0)

    def test_actual_thick_line_detection(self):
        image = np.zeros((84,84,3),np.uint8)
        cv2.line(image,(5,42),(5,53),(200,200,0),3)
        cv2.line(image,(5,63),(5,83),(200,200,0),3)
        cv2.line(image,(78,35),(78,83),(255,255,255),3)
        env = LineReward.__new__(LineReward)
        env._reward_image = image
        env.missing_steps = 0
        value,ended = env.reward_fn()
        self.assertGreater(value,.9)
        self.assertFalse(ended)
        self.assertEqual(env._left_line_type,"dashed")
        self.assertEqual(env._right_line_type,"solid")

    def test_cnn_dqn_learning_and_checkpoint(self):
        torch.set_num_threads(1)
        robot = MagicMock()
        robot.camera.capture.return_value = np.zeros((480,640,3),np.uint8)
        with patch("training.reward.DB21J",return_value=robot):
            env = DiscreteSteering(FrameStackCamera(RoadCamera(LineReward())),speed=.45)
        env.unwrapped.observation_interval = 0
        env.unwrapped.reset_settle_seconds = 0
        robot.pose.start.assert_not_called()
        robot.map_tiles.start.assert_not_called()
        try:
            model = DQN("CnnPolicy",env,device="auto",buffer_size=32,learning_starts=4,batch_size=4,
                        policy_kwargs={"features_extractor_class":DrivingCNN,"net_arch":[]})
            before = [p.detach().clone() for p in model.q_net.parameters()]
            model.learn(total_timesteps=24)
            self.assertGreater(model._n_updates,0)
            self.assertTrue(any(not torch.equal(old,new) for old,new in zip(before,model.q_net.parameters())))
            if torch.cuda.is_available():
                self.assertEqual(model.device.type,"cuda")
            model.model_version = MODEL_VERSION
            model.observation_color_space = "BINARY_PAINT"
            model.camera_settings = CAMERA_SETTINGS
            model.training_config = {"MAX_SPEED":.45,"LEARNING_RATE":.0003}
            with tempfile.TemporaryDirectory(prefix="duckie-simple-test-") as folder:
                path = Path(folder) / "model.zip"
                model.save(path)
                loaded = DQN.load(path,device="cpu")
                validate_model(loaded)
                observation,_ = env.reset()
                tensor,_ = loaded.policy.obs_to_tensor(observation)
                self.assertEqual(tuple(loaded.q_net(tensor).shape),(1,len(ACTION_NAMES)))
                self.assertEqual(loaded.observation_space.shape,(FRAME_COUNT,32,32))
            with self.assertRaises(ValueError):
                validate_model(SimpleNamespace(model_version="old"))
        finally:
            env.close()
        robot.motors.set_pwm.assert_called_with(left=0.,right=0.)

    def test_matplotlib_rgb_preview(self):
        import matplotlib
        matplotlib.use("Agg",force=True)
        import matplotlib.pyplot as plt
        viewer = CameraPreview()
        viewer.figure,axes = plt.subplots(1,2)
        image = np.zeros((84,84,3),np.uint8)
        viewer.plot = axes[0].imshow(image)
        viewer.detection_plot = axes[1].imshow(image)
        viewer.show(image)
        viewer._preview_key(SimpleNamespace(key="q"))
        with self.assertRaises(KeyboardInterrupt):
            viewer.show(image)
        viewer.close()


if __name__ == "__main__":
    unittest.main()
