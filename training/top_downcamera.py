"""Bird's-eye camera observations for the Duckiematrix DQN.

Insert between RoadCamera and FrameStackCamera. This changes only the CNN
observation; StripedRoadReward continues to see the original camera image.

Use a calibrated image-to-ground homography for geometrically meaningful output.
Without one, the four-point fallback is a VISUAL approximation, not calibration.
"""

import warnings

import cv2
import gymnasium as gym
import numpy as np


class TopDownCamera(gym.ObservationWrapper):
    """Warp HWC uint8 camera observations; optionally preview before and after."""

    # Normalized image corners ordered top-left, top-right, bottom-right, bottom-left.
    # These are starting guesses and MUST be checked against actual camera frames.
    DEFAULT_SOURCE_CORNERS = (
        (0.39, 0.52),
        (0.61, 0.52),
        (0.98, 0.98),
        (0.02, 0.98),
    )

    def __init__(
        self,
        env,
        *,
        homography=None,
        ground_bounds=(-0.35, 0.35, 0.05, 1.05),
        source_corners=None,
        output_size=None,
        camera_matrix=None,
        distortion_coefficients=None,
        preview=False,
        preview_every=5,
    ):
        """Initialize the projection.

        homography: optional 3x3 matrix mapping UNDISTORTED image pixels to
            ground coordinates (X lateral, Y forward), in metres. If supplied,
            `ground_bounds` determines the visible rectangle and orientation.
        ground_bounds: (left_x, right_x, near_y, far_y), metres, in the same
            ground coordinate frame as `homography`. Near is displayed at bottom.
        source_corners: optional four normalized (x,y) points for an UNCALIBRATED
            visual fallback. Ignored when `homography` is supplied.
        output_size: (width, height). Defaults to incoming image dimensions.
        camera_matrix/distortion_coefficients: optional camera intrinsics for
            undistortion, applied BEFORE the homography. Supply both or neither.
        preview: show original and warped CNN inputs side by side in Matplotlib.
        """
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 3:
            raise ValueError("TopDownCamera expects an HWC Box image observation")
        height, width, channels = space.shape
        if space.dtype != np.uint8 or channels not in (1, 3, 4):
            raise ValueError("TopDownCamera expects uint8 HWC images with 1, 3, or 4 channels")

        self.output_size = tuple(map(int, output_size or (width, height)))
        out_w, out_h = self.output_size
        if out_w <= 0 or out_h <= 0:
            raise ValueError("output_size must be positive")
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(out_h, out_w, channels), dtype=np.uint8
        )

        if (camera_matrix is None) != (distortion_coefficients is None):
            raise ValueError("Supply both camera_matrix and distortion_coefficients, or neither")
        self.camera_matrix = None if camera_matrix is None else np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.distortion_coefficients = None if distortion_coefficients is None else np.asarray(distortion_coefficients, dtype=np.float64).ravel()

        if homography is not None:
            H_image_to_ground = np.asarray(homography, dtype=np.float64)
            if H_image_to_ground.shape != (3, 3) or not np.isfinite(H_image_to_ground).all():
                raise ValueError("homography must be a finite 3x3 image-to-ground matrix")
            left, right, near, far = map(float, ground_bounds)
            if not (left < right and near < far):
                raise ValueError("ground_bounds must be (left < right, near < far)")
            # Map ground rectangle: far is the TOP of the rendered image.
            ground_corners = np.float32([
                [left, far], [right, far], [right, near], [left, near]
            ])
            image_corners = cv2.perspectiveTransform(
                ground_corners.reshape(1, 4, 2),
                np.linalg.inv(H_image_to_ground).astype(np.float64),
            ).reshape(4, 2).astype(np.float32)
            self.projection_mode = "calibrated"
        else:
            corners = source_corners or self.DEFAULT_SOURCE_CORNERS
            points = np.asarray(corners, dtype=np.float32)
            if points.shape != (4, 2) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
                raise ValueError("source_corners must contain 4 normalized (x, y) points")
            image_corners = points * np.array([width - 1, height - 1], dtype=np.float32)
            self.projection_mode = "approximate"
            warnings.warn(
                "TopDownCamera has no calibrated homography; using approximate "
                "source_corners. Output is NOT metric top-down projection.",
                RuntimeWarning,
                stacklevel=2,
            )

        dest_corners = np.float32([
            [0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]
        ])
        self.transform_matrix = cv2.getPerspectiveTransform(image_corners, dest_corners)
        if not np.isfinite(self.transform_matrix).all() or abs(np.linalg.det(self.transform_matrix)) < 1e-12:
            raise ValueError("The homography produces a degenerate image transform")

        self.preview = bool(preview)
        self.preview_every = max(1, int(preview_every))
        self._preview_count = 0
        self._figure = None
        self._original_artist = None
        self._warped_artist = None

    def observation(self, observation):
        source = observation
        if self.camera_matrix is not None:
            source = cv2.undistort(source, self.camera_matrix, self.distortion_coefficients)

        warped = cv2.warpPerspective(
            source, self.transform_matrix, self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if warped.ndim == 2:
            warped = warped[:, :, None]
        warped = np.ascontiguousarray(warped, dtype=np.uint8)

        self._preview_count += 1
        if self.preview and self._preview_count % self.preview_every == 0:
            self._update_preview(observation, warped)
        return warped

    @staticmethod
    def _for_plot(image):
        return image[:, :, 0] if image.ndim == 3 and image.shape[2] == 1 else image

    def _update_preview(self, original, warped):
        import matplotlib.pyplot as plt

        if self._figure is None or not plt.fignum_exists(self._figure.number):
            plt.ion()
            self._figure, axes = plt.subplots(1, 2, figsize=(11, 5), num="Duckiematrix: original vs top-down")
            self._original_artist = axes[0].imshow(self._for_plot(original))
            self._warped_artist = axes[1].imshow(self._for_plot(warped))
            axes[0].set_title("Original CNN camera observation")
            axes[1].set_title(f"Top-down CNN observation ({self.projection_mode})")
            for ax in axes:
                ax.axis("off")
            self._figure.tight_layout()
            plt.show(block=False)
        else:
            self._original_artist.set_data(self._for_plot(original))
            self._warped_artist.set_data(self._for_plot(warped))
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()

    def close(self):
        if self._figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self._figure)
            self._figure = None
        super().close()
