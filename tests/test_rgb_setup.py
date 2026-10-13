"""Offline tests of the five-action paper-inspired Duckiematrix pipeline."""

import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from stable_baselines3 import DQN
from training.environment import CAMERA_SETTINGS, MODEL_VERSION, CameraHistory, DiscreteSteering, FrameStackCamera, RoadCamera, camera_input, steering_wheels
from training.network import DrivingCNN
from training.reward import LaneGeometry, LaneReward
from training.train import TOTAL_TIMESTEPS
from utils.utils import validate_model, saved_speed, SaveRewards
from scripts.run_real import real_wheels
from scripts.visualizer import CameraPreview


def fake_robot():
    robot = MagicMock()
    robot.camera.capture.return_value = np.zeros((480,640,3), np.uint8)
    robot.pose.capture.return_value = pose(.117, 0., 1.)
    robot.map_tile_info.capture.return_value = {"data": {"map_0": {"tile_size": {"x": .585, "y": .585}}}}
    robot.map_tiles.capture.return_value = {"data": {"map_0/tile_0_0": {"type": "straight"},
                                                  "map_0/tile_0_1": {"type": "straight"}}}
    robot.map_frames.capture.return_value = {"data": {
        "map_0/tile_0_0": {"unit": "tiles", "pose": {"x": 0., "y": 0., "yaw": 0.}},
        "map_0/tile_0_1": {"unit": "tiles", "pose": {"x": 0., "y": 1., "yaw": 0.}},
    }}
    return robot


def pose(x, y, stamp, yaw=math.pi / 2):
    return {"position": {"x": x, "y": y},
            "rotation": {"w": math.cos(yaw / 2), "x": 0., "y": 0., "z": math.sin(yaw / 2)},
            "header": {"timestamp": stamp}}


class PaperPipelineTests(unittest.TestCase):
    def test_preprocessing(self):
        image = np.zeros((480,640,3), np.uint8)
        image[:, :320] = (255,255,0)
        image[:, 320:] = (255,255,255)
        processed = camera_input(image)
        self.assertEqual(processed.shape, (40,80,3))
        self.assertEqual(processed.dtype, np.uint8)
        self.assertEqual(processed[20,10].tolist(), [255,0,0])
        self.assertEqual(processed[20,70].tolist(), [0,255,0])
        self.assertTrue(np.all(processed[:,:,2] == 0))

    def test_five_actions_fixed_speed_budget(self):
        for action in range(5):
            wheels = steering_wheels(action,.45)
            self.assertTrue(np.all(wheels > 0))
            self.assertAlmostEqual(float(wheels.sum()),.45,places=6)
            self.assertLessEqual(float(wheels.max() / wheels.min()),4.000001)
        np.testing.assert_allclose(steering_wheels(2,.45),[.225,.225])
        np.testing.assert_allclose(steering_wheels(0,.45),[.36,.09])
        np.testing.assert_allclose(steering_wheels(4,.45),[.09,.36])
        with self.assertRaises(ValueError):
            steering_wheels(5,.45)

    def test_frame_history(self):
        history = CameraHistory()
        for value in range(6):
            stacked = history.push(np.full((40,80,3),value,np.uint8))
        self.assertEqual(stacked.shape,(40,80,15))
        self.assertEqual(stacked[0,0,::3].tolist(),[1,2,3,4,5])
        self.assertTrue(np.all(CameraHistory().push(np.full((40,80,3),9,np.uint8)) == 9))

    def test_geometry_right_wrong_lane_and_tile_seam(self):
        robot = fake_robot()
        data = {"frames": robot.map_frames.capture(), "tiles": robot.map_tiles.capture(),
                "tile_info": robot.map_tile_info.capture()}
        geometry = LaneGeometry(data,"map_0")
        right = geometry.measure(np.array([.117,0.]),np.array([0.,1.]))
        self.assertTrue(right["in_right_lane"])
        self.assertAlmostEqual(right["lane_distance"],0)
        self.assertAlmostEqual(right["heading_alignment"],1)
        wrong = geometry.measure(np.array([-.117,0.]),np.array([0.,1.]))
        self.assertFalse(wrong["in_right_lane"])
        self.assertFalse(wrong["off_road"])
        seam = geometry.measure(np.array([.117,.2925]),np.array([0.,1.]))
        self.assertTrue(seam["in_right_lane"])
        self.assertFalse(seam["off_road"])
        self.assertTrue(geometry.measure(np.array([1.,1.]),np.array([0.,1.]))["off_road"])

    def test_reward_formula(self):
        env = LaneReward.__new__(LaneReward)
        env._speed = .2
        env._lane = {"off_road": False, "in_right_lane": True, "lane_distance": .01, "heading_alignment": .8}
        reward, ended = env.reward_fn()
        self.assertAlmostEqual(reward,10 * .2 * .8 - 100 * .01)
        self.assertFalse(ended)
        env._lane["in_right_lane"] = False
        self.assertEqual(env.reward_fn(),(0.,False))
        env._lane["off_road"] = True
        self.assertEqual(env.reward_fn(),(-40.,True))

    def test_pose_speed_offroad_stop_and_time_limit(self):
        robot = fake_robot()
        with patch("training.reward.DB21J",return_value=robot):
            env = LaneReward()
        env.observation_interval = 0
        try:
            env.reset()
            robot.pose.capture.return_value = pose(.117,.01,1.05)
            _, reward, terminated, truncated, info = env.step([.225,.225])
            self.assertAlmostEqual(info["speed_m_s"],.2)
            self.assertAlmostEqual(reward,2.)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            env.max_episode_steps = 2
            robot.pose.capture.return_value = pose(.117,.02,1.10)
            self.assertTrue(env.step([.225,.225])[3])
            robot.motors.set_pwm.assert_called_with(left=0.,right=0.)
            robot.pose.capture.return_value = pose(1.,1.,1.15)
            _, reward, terminated, truncated, info = env.step([.225,.225])
            self.assertEqual(reward,-40.)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertIn("road boundary",info["termination_reason"])
        finally:
            env.close()

    def test_model_save_load_and_cuda_learning(self):
        torch.set_num_threads(1)
        robot = fake_robot()
        with patch("training.reward.DB21J",return_value=robot):
            env = DiscreteSteering(FrameStackCamera(RoadCamera(LaneReward())),speed=.45)
        env.unwrapped.observation_interval = 0
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = DQN("CnnPolicy",env,device=device,buffer_size=32,learning_starts=4,
                        batch_size=4,policy_kwargs={"features_extractor_class":DrivingCNN,"net_arch":[]})
            before = [value.detach().clone() for value in model.q_net.parameters()]
            model.learn(total_timesteps=16)
            self.assertGreater(model._n_updates,0)
            self.assertTrue(any(not torch.equal(old,new) for old,new in zip(before,model.q_net.parameters())))
            if torch.cuda.is_available():
                self.assertEqual(model.device.type,"cuda")
            model.model_version = MODEL_VERSION
            model.observation_color_space = "SEGMENTED_RGB"
            model.training_config = {"MAX_SPEED":.45}
            model.camera_settings = CAMERA_SETTINGS
            model.reward_settings = env.unwrapped.reward_settings()
            with tempfile.TemporaryDirectory(prefix="duckie-paper-test-") as folder:
                path = Path(folder) / "model.zip"
                model.save(path)
                loaded = DQN.load(path,device="cpu")
                validate_model(loaded)
                self.assertEqual(saved_speed(loaded),.45)
                observation,_ = env.reset()
                tensor,_ = loaded.policy.obs_to_tensor(observation)
                self.assertEqual(tuple(loaded.q_net(tensor).shape),(1,5))
                loaded.predict(observation,deterministic=True)
                self.assertEqual(loaded.reward_settings["off_road_penalty"],40.)
        finally:
            env.close()

    def test_model_validation_and_real_gain(self):
        with self.assertRaises(ValueError):
            validate_model(SimpleNamespace(model_version="old"))
        np.testing.assert_allclose(real_wheels(2,.45,1.5),[.3375,.3375])
        wheels = real_wheels(0,.45,10.)
        self.assertLessEqual(float(wheels.max()),1.)
        self.assertAlmostEqual(float(wheels[0] / wheels[1]),4.,places=6)

    def test_log_columns(self):
        episodes, batches = io.StringIO(), io.StringIO()
        callback = SaveRewards(episodes,batches)
        callback.locals = {"rewards": [1.], "dones": [True], "infos": [{"lane_distance":.01}]}
        callback._on_step()
        lines = episodes.getvalue().splitlines()
        self.assertEqual(len(lines[0].split(",")),len(lines[1].split(",")))
        self.assertIn("lane_distance",lines[0])
        self.assertEqual(TOTAL_TIMESTEPS,24000)

    def test_matplotlib(self):
        import matplotlib
        matplotlib.use("Agg",force=True)
        import matplotlib.pyplot as plt
        viewer = CameraPreview()
        viewer.figure, axes = plt.subplots(1,2)
        image = np.zeros((40,80,3),np.uint8)
        viewer.plot = axes[0].imshow(image)
        viewer.detection_plot = axes[1].imshow(image)
        viewer.show(image)
        viewer._preview_key(SimpleNamespace(key="q"))
        with self.assertRaises(KeyboardInterrupt):
            viewer.show(image)
        viewer.close()


if __name__ == "__main__":
    unittest.main()
