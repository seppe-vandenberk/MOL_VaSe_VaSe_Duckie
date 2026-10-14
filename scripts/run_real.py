"""Drive a real Duckiebot using a saved DQN model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import math
import time

from stable_baselines3 import DQN
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.robots.duckiebot.generic import DTPSMotorsDriver
from duckietown.sdk.middleware.dtps.components import GenericDTPSPublisher

from training.environment import camera_input, line_mask, steering_wheels, CameraHistory
from scripts.visualizer import CameraPreview
from utils.line_detection import LineDetector
from utils.utils import saved_speed, saved_steering_gain, validate_model


# Real-bot only: raise this to compensate for the motors' deadzone.
# Example: total speed 0.45 and gain 1.5 give straight PWM 0.3375 per wheel.
MOTOR_GAIN = 1.5


def real_wheels(action, speed, gain, steering_gain=.6):
    """Scale the model's wheel commands, preserving steering and forward motion."""
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("MOTOR_GAIN must be positive and finite")
    wheels = steering_wheels(action, speed, steering_gain) * gain
    # PWM cannot exceed 1. Scale both together to keep the steering ratio.
    maximum = float(wheels.max())
    if maximum > 1.0:
        wheels /= maximum
    return wheels


class RealMotors(DTPSMotorsDriver):
    """Send commands to the physical driver's input, not its filtered output."""

    def __init__(self, bot_name):
        GenericDTPSPublisher.__init__(
            self, f"{bot_name}.local", 11911, bot_name,
            ("actuator", "wheels", "base", "pwm"),
        )


def drive(model, bot_name, speed, duration, preview=True, gain=MOTOR_GAIN):
    """Camera -> CNN -> steering, without simulator reset or reward logic."""
    bot_name = bot_name.removesuffix(".local")
    robot = DB21J(bot_name, simulated=False)
    robot.add("motors", "base", RealMotors(bot_name))
    viewer = CameraPreview() if preview else None
    detector = LineDetector()
    history = CameraHistory()
    steering_gain = saved_steering_gain(model)
    try:
        robot.motors.start()
        robot.motors.set_pwm(left=0.0, right=0.0)
        robot.camera.start()
        # Wait for the first camera frame with the wheels stopped.
        frame = robot.camera.capture(block=True, timeout=10.0)
        if frame is None:
            raise RuntimeError("No camera frame received from the Duckiebot.")
        started = time.monotonic()
        last_frame = started
        while time.monotonic() - started < duration:
            image = camera_input(frame[:, :, [2, 1, 0]], model.camera_settings)
            observation = history.push(line_mask(image, model.camera_settings))
            if viewer is not None:
                detector.detect(image)
                viewer.show(image, detector, raw_image=frame[:, :, [2,1,0]],
                            cnn_input=observation)
            action, _ = model.predict(observation, deterministic=True)
            left, right = real_wheels(action, speed, gain, steering_gain)
            # Use the shared steering mapping with the compensated real speed.
            robot.motors.set_pwm(left=float(left), right=float(right))
            remaining = 0.05 - (time.monotonic() - last_frame)
            if remaining > 0:
                time.sleep(remaining)
            frame = robot.camera.capture(block=True, timeout=0.5)
            last_frame = time.monotonic()
            if frame is None:
                raise RuntimeError("Camera timed out; stopping the Duckiebot.")
    finally:
        try:
            robot.motors.set_pwm(left=0.0, right=0.0)
        finally:
            for component in (robot.camera, robot.motors):
                try:
                    component.stop()
                except Exception as error:
                    print(f"Component shutdown: {error}")
            if viewer is not None:
                viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the DQN model .zip")
    parser.add_argument("duckiebot", help="Duckiebot name, e.g. duckie01")
    parser.add_argument("--speed", type=float,
                        help="override the total wheel-speed budget, bypassing MOTOR_GAIN")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="driving time in seconds (default: 60)")
    parser.add_argument("--no-preview", action="store_true",
                        help="drive without the camera visualizer")
    args = parser.parse_args()
    if not args.model.is_file() or args.model.suffix.lower() != ".zip":
        parser.error("model must be an existing .zip file")
    if not 0 < args.duration < float("inf"):
        parser.error("--duration must be positive and finite")
    # Validate the model before connecting to the motors.
    model = DQN.load(args.model, device="auto")
    try:
        speed = saved_speed(model) if args.speed is None else args.speed
        validate_model(model)
        gain = MOTOR_GAIN if args.speed is None else 1.0
        # Validate commands before starting hardware; no model-saved gain
        # overrides the editable constant above.
        straight = real_wheels(2, speed, gain, saved_steering_gain(model))
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Driving {args.duckiebot} with {args.model}, speed={speed:g}, "
        f"motor_gain={gain:g}, straight PWM={float(straight[0]):g}. Ctrl+C stops."
    )
    try:
        preview = not args.no_preview
        drive(model, args.duckiebot, speed, args.duration, preview=preview, gain=gain)
    except KeyboardInterrupt:
        print("Stopped by driver.")
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
