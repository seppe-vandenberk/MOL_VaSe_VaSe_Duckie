"""Recovery, time-budget, checkpoint, and evaluation regression tests."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gymnasium as gym
import cv2
import numpy as np
import torch
from stable_baselines3 import DQN

from scripts.evaluate import evaluate_run, evaluate_runs
from training.callbacks import TrainingProgress
from training.environment import FRAME_COUNT, MODEL_VERSION, steering_wheels
from training.reward import LineReward
from training.train import build_model, main, parse_args
from utils.utils import saved_steering_gain
from utils.line_detection import LineDetector


class FakeClock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TestCamera(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (32, 32, FRAME_COUNT), np.uint8)
    action_space = gym.spaces.Discrete(5)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.steps = 0
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action):
        self.steps += 1
        image = np.zeros(self.observation_space.shape, np.uint8)
        image[:, 5:8] = 255
        image[:, 24:27] = 255
        return image, 1. if action == 2 else -.5, False, False, {"detection_state": "both"}


def reward_env():
    env = LineReward.__new__(LineReward)
    env._reward_image = np.zeros((32, 32, 3), np.uint8)
    env.missing_steps = env.partial_steps = env.episode_steps = 0
    env._lane_width = env._top_lane_width = None
    return env


def boundary(x, kind):
    return (10, x, (x, 31, x, 0), [], kind)


class RecoveryTests(unittest.TestCase):
    def test_boundary_identity_survives_crossing_image_center(self):
        detector = LineDetector()
        image = np.zeros((32, 32, 3), np.uint8)
        cv2.line(image, (5, 0), (5, 31), (255, 255, 255), 3)
        left, right = detector.detect(image)
        self.assertIsNone(left)
        self.assertIsNotNone(right)
        image[:] = 0
        cv2.line(image, (26, 0), (26, 31), (255, 255, 0), 3)
        left, right = detector.detect(image)
        self.assertIsNotNone(left)
        self.assertIsNone(right)

    def test_single_line_without_prior_geometry_eventually_fails(self):
        env = reward_env()
        env.detect = lambda _: (boundary(3, "dashed"), None)
        for i in range(env.max_partial_line_steps):
            reward, ended = env.reward_fn()
            self.assertEqual(ended, i == env.max_partial_line_steps - 1)
            self.assertIsNone(env._lane_center_error)
        self.assertLess(reward, -env.one_line_penalty)
        self.assertIn("not recovered", env.reason)

    def test_recent_width_guides_recovery_and_both_lines_clear_counters(self):
        env = reward_env()
        left, right = boundary(3, "dashed"), boundary(28, "solid")
        env.detect = lambda _: (left, right)
        self.assertEqual(env.reward_fn(), (1., False))
        env.detect = lambda _: (left, None)
        centered, _ = env.reward_fn()
        env.detect = lambda _: (boundary(10, "dashed"), None)
        displaced, _ = env.reward_fn()
        self.assertGreater(centered, displaced)
        self.assertGreater(env._lane_center_error, 0)
        env.detect = lambda _: (left, right)
        env.reward_fn()
        self.assertEqual((env.partial_steps, env.missing_steps), (0, 0))
        self.assertIsNone(env.reason)

    def test_alternating_one_and_zero_lines_cannot_avoid_failure(self):
        env = reward_env()
        for i in range(env.max_partial_line_steps):
            pair = (boundary(3, "dashed"), None) if i % 2 else (None, None)
            env.detect = lambda _, pair=pair: pair
            _, ended = env.reward_fn()
        self.assertTrue(ended)

    def test_invalid_pair_never_gets_positive_reward(self):
        env = reward_env()
        env.detect = lambda _: (boundary(28, "dashed"), boundary(3, "solid"))
        self.assertLess(env.reward_fn()[0], 0)
        self.assertIsNone(env._lane_width)

    def test_episode_limit_is_truncation_and_stops_motors(self):
        env = reward_env()
        env.robot = MagicMock()
        env.max_episode_steps = 1
        env._camera = lambda: np.zeros((32, 32, 3), np.uint8)
        env.detect = lambda _: (boundary(3, "dashed"), boundary(28, "solid"))
        env._left_line_type, env._right_line_type = "dashed", "solid"
        _, _, terminated, truncated, info = env.step([.175, .175])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "episode time limit")
        env.robot.motors.set_pwm.assert_called_with(left=0., right=0.)

    def test_camera_wait_is_included_in_frame_period(self):
        clock = FakeClock()
        env = reward_env()
        env.camera_settings = {"crop_top": .5}
        env.last_frame_time = None
        env.robot = MagicMock()
        requests = []
        def capture(**kwargs):
            requests.append(clock.now)
            clock.now += .02
            return np.zeros((480, 640, 3), np.uint8)
        env.robot.camera.capture.side_effect = capture
        with patch("training.reward.time.monotonic", clock.monotonic), patch("training.reward.time.sleep", clock.sleep):
            env._camera()
            env._camera()
        self.assertAlmostEqual(requests[1] - requests[0], .05)

    def test_steering_gain_preserves_old_models_and_allows_four_to_one(self):
        self.assertEqual(saved_steering_gain(SimpleNamespace(training_config={})), .6)
        old = steering_wheels(0, .35)
        wider = steering_wheels(0, .35, 1.)
        self.assertAlmostEqual(float(old[0] / old[1]), 2.125, places=5)
        self.assertAlmostEqual(float(wider[0] / wider[1]), 4., places=5)
        self.assertAlmostEqual(float(wider.sum()), .35, places=6)
        with self.assertRaises(ValueError):
            steering_wheels(-1, .35)


class BudgetTests(unittest.TestCase):
    def test_new_schedule_updates_weights_and_round_trips(self):
        torch.set_num_threads(1)
        model = build_model(TestCamera(), .0001, device="cpu")
        old = [p.detach().clone() for p in model.q_net.parameters()]
        model.learn(total_timesteps=520)
        self.assertEqual(model._n_updates, 20)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old, model.q_net.parameters())))
        model.model_version = MODEL_VERSION
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.zip"
            model.save(path)
            loaded = DQN.load(path, device="cpu")
            warm = build_model(TestCamera(), .0001, device="cpu", warm_start=loaded)
            for a, b in zip(model.q_net.parameters(), warm.q_net.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(warm.replay_buffer.size(), 0)

    def test_expired_budget_stops_and_logs(self):
        model = build_model(TestCamera(), .0001, device="cpu")
        stream = io.StringIO()
        callback = TrainingProgress(stream, deadline=-1)
        with contextlib.redirect_stdout(io.StringIO()):
            model.learn(total_timesteps=1000, callback=callback)
        self.assertEqual(model.num_timesteps, 1)
        self.assertEqual(callback.stop_reason, "wall-clock limit")
        self.assertIn("reward_per_step", stream.getvalue())
        self.assertEqual(len(stream.getvalue().splitlines()), 2)

    def test_invalid_budgets_are_rejected_before_connecting(self):
        for args in (["--train-minutes", "31"], ["--steps", "0"], ["--budget-minutes", "nan"], ["--learning-rate", "nan"], ["--learning-rate", "0"], ["--speed", "nan"], ["--speed", "0"], ["--port-offset", "48019"], ["--port-offset", "-1"], ["--run-name", "../outside"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)

    def test_launcher_overrides_route_to_offset_and_save_actual_steps(self):
        robot = MagicMock()
        robot.camera.capture.return_value = np.zeros((480, 640, 3), np.uint8)
        run_name = "dqn_simple_test_lr0.0002_speed0.4_offset20"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.yaml").write_text("MAX_SPEED: 0.35\nLEARNING_RATE: 0.0001\n")
            with patch("training.train.PROJECT_ROOT", root), \
                 patch("training.reward.DB21J", return_value=robot) as factory, \
                 patch.object(LineReward, "reset_settle_seconds", 0), \
                 patch.object(LineReward, "observation_interval", 0), \
                 patch("training.train.TrainingProgress", side_effect=lambda stream, deadline: TrainingProgress(stream, -1)), \
                 contextlib.redirect_stdout(io.StringIO()):
                main(["--steps", "40", "--eval-runs", "0", "--device", "cpu", "--no-preview",
                      "--speed", ".4", "--learning-rate", ".0002", "--port-offset", "20", "--run-name", run_name])
            factory.assert_called_once_with("map_0/vehicle_0", simulated=True, port=7521)
            result = json.loads((root / "training/logs" / run_name / "summary.json").read_text())
            self.assertEqual(result['steps'], 1)
            self.assertEqual(Path(result['model']).name, run_name + "_final_1_steps.zip")
            loaded = DQN.load(result['model'], device="cpu")
            self.assertEqual(loaded.training_config['MAX_SPEED'], .4)
            self.assertEqual(loaded.training_config['LEARNING_RATE'], .0002)
            self.assertEqual(loaded.simulator_port_offset, 20)
            self.assertEqual((root / "config.yaml").read_text(), "MAX_SPEED: 0.35\nLEARNING_RATE: 0.0001\n")

    def test_training_cli_saves_logs_checkpoint_and_summary(self):
        robot = MagicMock()
        image = np.zeros((480, 640, 3), np.uint8)
        image[240:, 60:100] = 255
        image[240:, 540:580] = 255
        robot.camera.capture.return_value = image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.yaml").write_text("MAX_SPEED: 0.35\nLEARNING_RATE: 0.0001\n")
            with patch("training.train.PROJECT_ROOT", root), \
                 patch("training.reward.DB21J", return_value=robot), \
                 patch.object(LineReward, "reset_settle_seconds", 0), \
                 patch.object(LineReward, "observation_interval", 0), \
                 contextlib.redirect_stdout(io.StringIO()):
                main(["--steps", "12", "--eval-runs", "0", "--device", "cpu", "--no-preview"])
            summaries = list(root.glob("training/logs/*/summary.json"))
            self.assertEqual(len(summaries), 1)
            result = json.loads(summaries[0].read_text())
            self.assertEqual(result["steps"], 12)
            self.assertTrue(Path(result["model"]).is_file())
            loaded = DQN.load(result["model"], device="cpu")
            self.assertTrue(loaded.camera_settings["floor_filter"]["enabled"])
            self.assertEqual(loaded.camera_settings["floor_filter"]["excluded_floor_hsv_lower"], [130, 20, 0])
            self.assertEqual(loaded.observation_projection, "camera_crop_hsv_paint_floor_v2")
            self.assertTrue((summaries[0].parent / "progress.csv").is_file())
            self.assertIsNone(result["evaluation"])
        robot.motors.set_pwm.assert_called_with(left=0., right=0.)


class EvaluationTests(unittest.TestCase):
    def make_env(self, clock, *, terminated=False, truncated=False):
        env = MagicMock()
        env.reset.return_value = (np.zeros((32, 32, 1), np.uint8), {})
        def step(action):
            clock.now += .05
            return None, 1., terminated, truncated, dict(detection_state="both", lane_center_error=.1)
        env.step.side_effect = step
        return env

    def test_time_limits_are_not_failures(self):
        clock = FakeClock()
        env = self.make_env(clock, truncated=True)
        model = MagicMock()
        model.predict.return_value = (2, None)
        with patch("scripts.evaluate.time.monotonic", clock.monotonic), contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_run(model, env, 1.)
        self.assertEqual(result["failures"], 0)
        self.assertGreater(result["time_limits"], 0)
        self.assertTrue(result["passed"])
        self.assertTrue(all(call.kwargs["deterministic"] for call in model.predict.call_args_list))
        env.unwrapped.robot.motors.set_pwm.assert_called_with(left=0., right=0.)

    def test_failures_and_shortened_runs_cannot_pass(self):
        for failure, deadline in [(True, None), (False, .25)]:
            clock = FakeClock()
            env = self.make_env(clock, terminated=failure)
            model = MagicMock()
            model.predict.return_value = (2, None)
            with patch("scripts.evaluate.time.monotonic", clock.monotonic), contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_run(model, env, 1., deadline=deadline)
            self.assertFalse(result["passed"])
            if failure:
                self.assertGreater(result["failures"], 0)
            else:
                self.assertFalse(result["completed"])

    def test_evaluation_uses_different_initial_steering(self):
        clock = FakeClock()
        env = self.make_env(clock)
        model = MagicMock()
        model.predict.return_value = (2, None)
        with patch("scripts.evaluate.time.monotonic", clock.monotonic), contextlib.redirect_stdout(io.StringIO()):
            report = evaluate_runs(model, env, runs=3, duration=.5)
        self.assertEqual([r["start_action"] for r in report["runs"]], ["straight", "slight_right", "slight_left"])
        self.assertEqual(report["runs"][1]["actions"]["slight_right"], 5)


if __name__ == "__main__":
    unittest.main()
