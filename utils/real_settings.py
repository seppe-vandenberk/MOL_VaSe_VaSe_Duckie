"""Shared real-camera framing and steering calibration controls."""

import math

# Real camera: discard the top 40%, retaining the lower 60% of the image.
REAL_CROP_TOP = .40
# Full turns use a 4:1 outer/inner wheel ratio; total PWM remains constant.
REAL_STEERING_GAIN = 1.0


def real_camera_settings(settings, crop_top=REAL_CROP_TOP):
    """Apply real-camera framing without changing the checkpoint's settings."""
    if not math.isfinite(crop_top) or not 0 <= crop_top < 1:
        raise ValueError("crop-top must be finite and between 0 (inclusive) and 1 (exclusive)")
    copied = dict(settings)
    copied['crop_top'] = float(crop_top)
    return copied
