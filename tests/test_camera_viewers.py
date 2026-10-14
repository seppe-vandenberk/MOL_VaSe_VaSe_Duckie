"""Camera-only port routing and evaluation after baseline cleanup."""

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import evaluate, view_sim
from scripts.view_real import show_camera
from training.environment import CAMERA_SETTINGS, camera_input, line_mask
from utils.real_settings import real_camera_settings
import numpy as np


class CameraViewerTests(unittest.TestCase):
    def test_simulator_viewer_routes_selected_port_without_driving(self):
        model=SimpleNamespace(camera_settings=CAMERA_SETTINGS)
        with patch('scripts.view_sim.DQN.load', return_value=model), \
             patch('scripts.view_sim.validate_model'), \
             patch('scripts.view_sim.wait_for_camera') as wait, \
             patch('scripts.view_sim.show_camera') as show:
            view_sim.main(['--port', '7541'])
        wait.assert_called_once_with('map_0/vehicle_0', 7541, 10.)
        self.assertEqual(show.call_args.args, ('map_0/vehicle_0',))
        self.assertTrue(show.call_args.kwargs['simulated'])
        self.assertEqual(show.call_args.kwargs['port'], 7541)
        self.assertTrue(show.call_args.kwargs['settings']['floor_filter']['enabled'])

    def test_shared_viewer_uses_only_camera_and_cleans_up_on_timeout(self):
        for simulated, name, port in ((True, 'map_0/vehicle_0', 7541), (False, 'ducksolo.local', 11911)):
            with self.subTest(simulated=simulated):
                robot=MagicMock(); robot.camera.capture.return_value=None
                with patch('scripts.view_real.DB21J', return_value=robot) as factory, \
                     patch('scripts.view_real.CameraPreview') as viewer, \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'Camera timed out'):
                        show_camera(name, simulated=simulated, port=port)
                factory.assert_called_once_with(name if simulated else 'ducksolo', simulated=simulated, port=port)
                robot.motors.start.assert_not_called()
                robot.motors.set_pwm.assert_not_called()
                robot.reset_flag.set_reset.assert_not_called()
                robot.camera.stop.assert_called_once()
                viewer.return_value.close.assert_called_once()

    def test_real_preview_uses_the_same_wider_mask_as_driver(self):
        robot=MagicMock()
        frame=np.zeros((480,640,3),np.uint8);frame[200:230,80:120]=255
        robot.camera.capture.return_value=frame
        with patch('scripts.view_real.DB21J',return_value=robot), \
             patch('scripts.view_real.CameraPreview') as viewer, \
             contextlib.redirect_stdout(io.StringIO()):
            viewer.return_value.show.side_effect=KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                show_camera('ducksolo.local',settings=CAMERA_SETTINGS)
        expected=camera_input(frame[:,:,::-1],real_camera_settings(CAMERA_SETTINGS))
        np.testing.assert_array_equal(viewer.return_value.show.call_args.args[0],expected)
        np.testing.assert_array_equal(viewer.return_value.show.call_args.kwargs['cnn_input'],line_mask(expected))
        robot.motors.start.assert_not_called()
        self.assertEqual(CAMERA_SETTINGS['crop_top'],.5)

    def test_evaluation_selects_renamed_baseline_and_selected_port(self):
        model=SimpleNamespace(camera_settings=CAMERA_SETTINGS, training_config={'MAX_SPEED':.35})
        reward=MagicMock(); reward.reward_version='test'; reward.reward_settings.return_value={}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'models').mkdir(); baseline=root/'models/baseline.zip'; baseline.touch()
            with patch.object(evaluate, 'PROJECT_ROOT', root), \
                 patch('sys.argv', ['evaluate.py', '--port', '7541']), \
                 patch.object(evaluate.DQN, 'load', return_value=model) as load, \
                 patch.object(evaluate, 'validate_model'), \
                 patch.object(evaluate, 'LineReward', return_value=reward) as env, \
                 patch.object(evaluate, 'RoadCamera'), patch.object(evaluate, 'FrameStackCamera'), \
                 patch.object(evaluate, 'DiscreteSteering'), \
                 patch.object(evaluate, 'evaluate_runs', return_value={'passed':True}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(), 0)
            load.assert_called_once_with(baseline, device='auto')
            env.assert_called_once_with(entity_name='map_0/vehicle_0', camera_settings=CAMERA_SETTINGS, port=7541)


if __name__ == '__main__':
    unittest.main()
