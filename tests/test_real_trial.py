"""Physical-driver stopping and camera-guidance checks with mocked hardware."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from scripts.run_real import LaneVisibilityGuard, drive, real_wheels
from training.environment import CAMERA_SETTINGS, camera_input, line_mask
from utils.real_settings import REAL_CROP_TOP, REAL_STEERING_GAIN, real_camera_settings


class RealTrialTests(unittest.TestCase):
    def frame(self):
        image = np.zeros((480, 640, 3), np.uint8)
        image[:, 80:120] = (0, 255, 255)
        image[:, 520:560] = (255, 255, 255)
        return image

    def setup(self):
        robot = MagicMock()
        model = MagicMock()
        model.camera_settings = CAMERA_SETTINGS
        model.training_config = {'STEERING_GAIN': .6}
        model.predict.return_value = (2, None)
        return robot, model

    def test_wider_crop_retains_additional_visible_paint_without_mutating_model(self):
        settings=real_camera_settings(CAMERA_SETTINGS)
        self.assertEqual(settings['crop_top'], .4)
        self.assertEqual(CAMERA_SETTINGS['crop_top'], .5)
        image=np.zeros((480,640,3),np.uint8)
        image[200:230,80:120]=255
        expected=cv2.resize(image[192:], (32,32), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(camera_input(image, settings), expected)
        self.assertEqual(np.count_nonzero(line_mask(camera_input(image))), 0)
        self.assertGreater(np.count_nonzero(line_mask(camera_input(image, settings))), 0)

    def test_stronger_turns_keep_total_power_and_straight_command(self):
        for action in range(5):
            old=real_wheels(action,.35,1.4,.6)
            new=real_wheels(action,.35,1.4)
            self.assertAlmostEqual(float(old.sum()), float(new.sum()), places=6)
            self.assertAlmostEqual(float(new.sum()), .49, places=6)
            self.assertTrue(np.all(new>0))
            if action==2:
                np.testing.assert_array_equal(old,new)
            else:
                self.assertEqual(np.sign(old[0]-old[1]), np.sign(new[0]-new[1]))
                self.assertGreater(abs(float(new[0]-new[1])), abs(float(old[0]-old[1])))
        full=real_wheels(0,.35,1.4)
        np.testing.assert_allclose(full, [.392,.098], atol=1e-7)
        self.assertAlmostEqual(float(full[0]/full[1]), 4., places=5)

    def test_motor_saturation_preserves_turn_ratio(self):
        wheels=real_wheels(0,1.,10.)
        self.assertLessEqual(float(wheels.max()),1.)
        self.assertAlmostEqual(float(wheels[0]/wheels[1]),4.,places=5)

    def test_driver_supplies_wider_mask_and_stronger_wheel_command(self):
        robot,model=self.setup(); frame=self.frame()
        robot.camera.capture.side_effect=[frame,None]
        model.predict.return_value=(0,None)
        with patch('scripts.run_real.DB21J',return_value=robot), \
             patch('scripts.run_real.RealMotors'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'Camera timed out'):
                drive(model,'testbot',.35,.5,preview=False,gain=1.4)
        expected=line_mask(camera_input(frame[:,:,::-1],real_camera_settings(CAMERA_SETTINGS)))
        np.testing.assert_array_equal(model.predict.call_args.args[0],expected)
        actual=robot.motors.set_pwm.call_args_list[1].kwargs
        np.testing.assert_allclose([actual['left'],actual['right']],[.392,.098],atol=1e-7)
        self.assertEqual(model.camera_settings['crop_top'],.5)
        robot.motors.set_pwm.assert_called_with(left=0.,right=0.)

    def test_invalid_calibration_is_rejected_before_connecting_hardware(self):
        robot,model=self.setup()
        for options in ({'crop_top':1.},{'crop_top':float('nan')},{'steering_gain':0.},{'steering_gain':1.1}):
            with patch('scripts.run_real.DB21J') as factory:
                with self.assertRaises(ValueError):
                    drive(model,'testbot',.35,.5,preview=False,**options)
                factory.assert_not_called()

    def test_single_boundary_survives_old_twenty_frame_limit(self):
        guard = LaneVisibilityGuard()
        for i in range(50):
            self.assertIsNone(guard.update(None, object(), now=i / 20))
        self.assertIn("Recovering yellow left", guard.status)
        self.assertEqual(guard.update(None, object(), now=3.), 'lane boundaries not recovered')

    def test_missing_boundaries_stop_by_time_and_recovery_resets(self):
        guard = LaneVisibilityGuard()
        self.assertIsNone(guard.update(None, None, now=0.))
        self.assertIsNone(guard.update(None, None, now=.49))
        self.assertEqual(guard.update(None, None, now=.5), 'both lane boundaries lost')
        self.assertIsNone(guard.update(object(), object(), now=.6))
        self.assertEqual((guard.missing_since, guard.partial_since), (None, None))
        self.assertIsNone(guard.update(object(), None, now=10.))
        self.assertIsNone(guard.update(object(), None, now=12.9))

    def test_camera_rate_does_not_change_single_boundary_deadline(self):
        for fps in (10, 20, 60):
            guard = LaneVisibilityGuard(single_boundary_timeout=2.)
            for i in range(fps * 2):
                self.assertIsNone(guard.update(object(), None, now=i / fps))
            self.assertEqual(guard.update(object(), None, now=2.), 'lane boundaries not recovered')

    def test_brief_total_loss_resets_missing_timer_but_not_recovery_window(self):
        guard = LaneVisibilityGuard()
        self.assertIsNone(guard.update(object(), None, now=0.))
        self.assertIsNone(guard.update(None, None, now=.1))
        self.assertIsNone(guard.update(None, object(), now=.4))
        self.assertIsNone(guard.update(None, None, now=2.8))
        self.assertEqual(guard.update(None, object(), now=3.), 'lane boundaries not recovered')

    def test_invalid_timeouts_are_rejected(self):
        for value in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                LaneVisibilityGuard(single_boundary_timeout=value)
            with self.assertRaises(ValueError):
                LaneVisibilityGuard(no_boundary_timeout=value)

    def test_visible_outer_edge_can_recover_after_more_than_twenty_frames(self):
        robot, model = self.setup()
        full = self.frame()
        outer_only = full.copy()
        outer_only[:, 80:120] = 0
        frames = iter([full] + [outer_only] * 25 + [full, None])
        clock = [0.]
        def capture(**kwargs):
            clock[0] += .05
            return next(frames)
        robot.camera.capture.side_effect = capture
        with patch('scripts.run_real.DB21J', return_value=robot), \
             patch('scripts.run_real.RealMotors'), \
             patch('scripts.run_real.time.monotonic', side_effect=lambda: clock[0]), \
             patch('scripts.run_real.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Camera timed out'):
                drive(model, 'testbot', .35, 5., preview=False)
        self.assertEqual(model.predict.call_count, 27)
        robot.motors.set_pwm.assert_called_with(left=0., right=0.)

    def test_sustained_single_boundary_still_stops_motors(self):
        robot, model = self.setup()
        full = self.frame()
        outer_only = full.copy()
        outer_only[:, 80:120] = 0
        clock = [0.]
        calls = [0]
        def capture(**kwargs):
            clock[0] += .1
            calls[0] += 1
            return full if calls[0] == 1 else outer_only
        robot.camera.capture.side_effect = capture
        with patch('scripts.run_real.DB21J', return_value=robot), \
             patch('scripts.run_real.RealMotors'), \
             patch('scripts.run_real.time.monotonic', side_effect=lambda: clock[0]), \
             patch('scripts.run_real.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
            report = drive(model, 'testbot', .35, 10., preview=False)
        self.assertEqual(report['stop_reason'], 'lane boundaries not recovered')
        self.assertGreaterEqual(report['seconds'], 3.)
        self.assertLess(report['seconds'], 3.3)
        self.assertFalse(report['completed'])
        self.assertTrue(any('Recovering yellow left' in sample['guidance_status'] for sample in report['samples']))
        robot.motors.set_pwm.assert_called_with(left=0., right=0.)

    def test_camera_timeout_stops_motors_after_a_valid_command(self):
        robot, model = self.setup()
        robot.camera.capture.side_effect = [self.frame(), None]
        with patch('scripts.run_real.DB21J', return_value=robot), \
             patch('scripts.run_real.RealMotors'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Camera timed out'):
                drive(model, 'testbot', .35, .5, preview=False)
        model.predict.assert_called_once()
        robot.motors.set_pwm.assert_called_with(left=0., right=0.)
        robot.motors.stop.assert_called_once()

    def test_initial_failure_is_shown_before_viewer_closes(self):
        robot, model = self.setup()
        robot.camera.capture.return_value = np.zeros((480, 640, 3), np.uint8)
        with patch('scripts.run_real.DB21J', return_value=robot), \
             patch('scripts.run_real.RealMotors'), \
             patch('scripts.run_real.CameraPreview') as viewer, contextlib.redirect_stdout(io.StringIO()):
            drive(model, 'testbot', .35, .5, preview=True)
        viewer.return_value.show.assert_called_once()
        detector = viewer.return_value.show.call_args.args[1]
        self.assertIn('STOP: both boundaries must be visible', detector._guidance_status)
        model.predict.assert_not_called()
        robot.motors.set_pwm.assert_called_with(left=0., right=0.)

    def test_missing_initial_lane_never_sends_nonzero_pwm(self):
        robot, model = self.setup()
        robot.camera.capture.return_value = np.zeros((480, 640, 3), np.uint8)
        with patch('scripts.run_real.DB21J', return_value=robot), \
             patch('scripts.run_real.RealMotors'), contextlib.redirect_stdout(io.StringIO()):
            report = drive(model, 'testbot', .35, .5, preview=False)
        self.assertFalse(report['completed'])
        model.predict.assert_not_called()
        for call in robot.motors.set_pwm.call_args_list:
            self.assertEqual(call.kwargs, dict(left=0., right=0.))


if __name__ == '__main__':
    unittest.main()
