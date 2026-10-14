"""Regression checks for the reduced camera input and its small CNN."""

import unittest
import cv2
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from training.environment import camera_input, line_mask, CameraHistory, FRAME_COUNT, MODEL_VERSION, ACTION_NAMES
from training.network import DrivingCNN
from utils.utils import validate_model


class Camera32Tests(unittest.TestCase):
    def test_exact_resize_and_mask(self):
        image = np.random.default_rng(0).integers(0,256,(480,640,3),dtype=np.uint8)
        cropped = camera_input(image)
        np.testing.assert_array_equal(cropped,cv2.resize(image[240:],(32,32),interpolation=cv2.INTER_AREA))
        mask = line_mask(cropped)
        self.assertEqual(mask.shape,(32,32,1))
        self.assertEqual(CameraHistory().push(mask).shape,(32,32,FRAME_COUNT))

    def test_small_cnn_dqn_learns(self):
        class FakeEnv(gym.Env):
            observation_space = gym.spaces.Box(0,255,(32,32,FRAME_COUNT),np.uint8)
            action_space = gym.spaces.Discrete(len(ACTION_NAMES))
            def reset(self, **kwargs):
                return np.zeros(self.observation_space.shape,np.uint8),{}
            def step(self, action):
                return np.zeros(self.observation_space.shape,np.uint8),1.,False,False,{}
        torch.set_num_threads(1)
        model = DQN("CnnPolicy",FakeEnv(),device="auto",buffer_size=32,learning_starts=4,batch_size=4,
                    policy_kwargs={"features_extractor_class":DrivingCNN,"net_arch":[]})
        model.model_version = MODEL_VERSION
        validate_model(model)
        model.learn(total_timesteps=16)
        self.assertGreater(model._n_updates,0)
        image = np.zeros((32,32,FRAME_COUNT),np.uint8)
        tensor,_ = model.policy.obs_to_tensor(image)
        self.assertEqual(tuple(model.q_net(tensor).shape),(1,len(ACTION_NAMES)))


if __name__ == "__main__":
    unittest.main()
