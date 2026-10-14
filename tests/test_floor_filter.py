"""Pale-floor rejection and shared training/deployment preprocessing checks."""

import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch
import queue
import numpy as np
import cv2

from training.environment import CAMERA_SETTINGS, line_mask, RoadCamera, FrameStackCamera, FRAME_COUNT
from utils.paint_mask import floor_camera_settings
from utils.line_detection import LineDetector
from scripts.visualizer import AsyncCameraPreview


class FloorFilterTests(unittest.TestCase):
    def setUp(self):
        self.settings = floor_camera_settings(CAMERA_SETTINGS, enabled=True)

    def road_on_pale_floor(self):
        image = np.full((32, 32, 3), 220, np.uint8)
        image[:, 9:27] = 35
        image[:, 12:14] = (255, 255, 0)
        image[:, 23:25] = 255
        return image

    def test_washed_out_purple_carpet_is_removed_and_tape_is_preserved(self):
        image = self.road_on_pale_floor()
        image[:, :9] = (200, 155, 200)
        image[:, 27:] = (200, 155, 200)
        self.assertTrue(np.all(line_mask(image)[:, :9] == 255))
        mask = line_mask(image, self.settings)
        self.assertTrue(np.all(mask[:, :9] == 0))
        self.assertTrue(np.all(mask[:, 27:] == 0))
        self.assertTrue(np.all(mask[:, 12:14] == 255))
        self.assertTrue(np.all(mask[:, 23:25] == 255))

    def test_blue_tint_on_dark_mat_is_not_classified_as_purple_carpet(self):
        image = self.road_on_pale_floor()
        road = np.all(image == 35, axis=2)
        image[road] = (40, 40, 60)
        mask = line_mask(image, self.settings)
        self.assertTrue(np.all(mask[:, 12:14] == 255))
        self.assertTrue(np.all(mask[:, 23:25] == 255))

    def test_desaturated_purple_is_removed_without_removing_neutral_tape(self):
        image = np.full((32, 32, 3), 35, np.uint8)
        image[:, 8:10] = (180, 165, 180)
        image[:, 22:24] = (255, 250, 255)
        self.assertTrue(np.all(line_mask(image)[:, 8:10] == 255))
        mask = line_mask(image, self.settings)
        self.assertTrue(np.all(mask[:, 8:10] == 0))
        self.assertTrue(np.all(mask[:, 22:24] == 255))

    def test_dark_purple_carpet_cannot_be_selected_as_main_road(self):
        image = self.road_on_pale_floor()
        image[:, :9] = (90, 60, 90)
        image[:, 27:] = (90, 60, 90)
        mask = line_mask(image, self.settings)
        self.assertTrue(np.all(mask[:, 12:14] == 255))
        self.assertTrue(np.all(mask[:, 23:25] == 255))

    def test_near_white_tape_wider_than_old_limit_is_preserved(self):
        image = np.full((32, 32, 3), 35, np.uint8)
        image[:, 12:22] = 255
        self.assertTrue(np.all(line_mask(image, self.settings)[:, 12:22] == 255))

    def test_nearby_diagonal_tape_is_not_rejected_by_horizontal_width(self):
        image = np.full((32, 32, 3), 35, np.uint8)
        cv2.line(image, (1, 5), (29, 29), (255, 255, 255), 5)
        self.assertGreater(np.count_nonzero(line_mask(image, self.settings)), 0)
        np.testing.assert_array_equal(line_mask(image), line_mask(image, self.settings))

    def test_uniform_white_floor_is_not_paint(self):
        image = np.full((32, 32, 3), 220, np.uint8)
        self.assertTrue(np.all(line_mask(image) == 255))
        self.assertTrue(np.all(line_mask(image, self.settings) == 0))

    def test_floor_is_removed_and_real_paint_is_preserved(self):
        image = self.road_on_pale_floor()
        mask = line_mask(image, self.settings)
        self.assertTrue(np.all(mask[:, :9] == 0))
        self.assertTrue(np.all(mask[:, 27:] == 0))
        self.assertTrue(np.all(mask[:, 12:14] == 255))
        self.assertTrue(np.all(mask[:, 23:25] == 255))
        self.assertEqual(mask.shape, (32, 32, 1))
        self.assertEqual(mask.dtype, np.uint8)

    def test_reward_and_policy_use_the_same_filtered_paint(self):
        image = self.road_on_pale_floor()
        detector = LineDetector()
        detector.camera_settings = self.settings
        left, right = detector.detect(image)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertLess(right[1], 27)
        np.testing.assert_array_equal(detector._paint_mask, line_mask(image, self.settings)[:, :, 0])

    def test_broad_bright_patch_inside_road_is_rejected(self):
        image = np.full((32, 32, 3), 35, np.uint8)
        image[8:24, 8:24] = 220
        self.assertEqual(np.count_nonzero(line_mask(image, self.settings)), 0)

    def test_bent_narrow_paint_is_kept_even_when_bbox_is_wide(self):
        image = np.full((32, 32, 3), 35, np.uint8)
        for row in range(32):
            x = 4 + row // 2
            image[row, x:x+2] = 255
        self.assertEqual(np.count_nonzero(line_mask(image, self.settings)), 64)

    def test_old_settings_remain_compatible_and_override_is_a_copy(self):
        image = self.road_on_pale_floor()
        old = dict(CAMERA_SETTINGS)
        new = floor_camera_settings(old, enabled=True, road_value_max=100)
        self.assertNotIn('floor_filter', old)
        self.assertTrue(new['floor_filter']['enabled'])
        copied = floor_camera_settings(new, enabled=False)
        self.assertTrue(new['floor_filter']['enabled'])
        np.testing.assert_array_equal(line_mask(image, old), line_mask(image, copied))

    def test_road_threshold_can_be_calibrated_for_gray_mat(self):
        image = self.road_on_pale_floor()
        image[:, 9:27][np.all(image[:, 9:27] == 35, axis=2)] = 115
        calibrated = floor_camera_settings(self.settings, road_value_max=118)
        self.assertEqual(np.count_nonzero(line_mask(image, self.settings)), 0)
        self.assertGreater(np.count_nonzero(line_mask(image, calibrated)), 0)


class AsyncViewerTests(unittest.TestCase):
    def test_queued_snapshot_matches_policy_input_and_cannot_be_mutated(self):
        import multiprocessing
        context = MagicMock()
        context.Queue.return_value = queue.Queue(maxsize=1)
        context.Event.return_value = MagicMock()
        context.Event.return_value.is_set.return_value = False
        with patch.object(multiprocessing, 'get_context', return_value=context):
            viewer = AsyncCameraPreview()
        crop = np.zeros((32, 32, 3), np.uint8)
        raw = np.zeros((480, 640, 3), np.uint8)
        mask = np.full((32, 32, FRAME_COUNT), 255, np.uint8)
        with patch('scripts.visualizer.time.monotonic', return_value=1.):
            viewer.show(crop, raw_image=raw, cnn_input=mask)
        _, _, snapshot_raw, snapshot_mask = viewer.frames.get_nowait()
        np.testing.assert_array_equal(snapshot_mask, mask)
        self.assertFalse(np.shares_memory(snapshot_mask, mask))
        mask[:] = 0
        raw[:] = 255
        self.assertTrue(np.all(snapshot_mask == 255))
        self.assertTrue(np.all(snapshot_raw == 0))


if __name__ == '__main__':
    unittest.main()
