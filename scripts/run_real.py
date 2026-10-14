"""Drive a real Duckiebot using a saved DQN model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import cv2
import math
import time

from stable_baselines3 import DQN
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.robots.duckiebot.generic import DTPSMotorsDriver
from duckietown.sdk.middleware.dtps.components import GenericDTPSPublisher

from training.environment import camera_input, line_mask, steering_wheels, CameraHistory, ACTION_NAMES
from scripts.visualizer import CameraPreview
from utils.line_detection import LineDetector
from utils.utils import saved_speed, validate_model
from utils.paint_mask import floor_camera_settings
from utils.real_settings import REAL_CROP_TOP, REAL_STEERING_GAIN, real_camera_settings


# Real-bot only: raise this to compensate for the motors' deadzone.
# Example: total speed 0.45 and gain 1.5 give straight PWM 0.3375 per wheel.
MOTOR_GAIN = 1.4


def real_wheels(action, speed, gain, steering_gain=REAL_STEERING_GAIN):
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


class LaneVisibilityGuard:
    """Allow timed single-boundary recovery, stop promptly when both disappear."""
    def __init__(self, no_boundary_timeout=.5, single_boundary_timeout=3.):
        for value in (no_boundary_timeout, single_boundary_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Boundary timeouts must be positive and finite")
        self.no_boundary_timeout = no_boundary_timeout
        self.single_boundary_timeout = single_boundary_timeout
        self.missing_since = self.partial_since = None
        self.status = "Both boundaries visible"

    def update(self, left, right, *, now=None):
        now = time.monotonic() if now is None else now
        if left is not None and right is not None:
            self.missing_since = self.partial_since = None
            self.status = "Both boundaries visible"
            return None
        if self.partial_since is None:
            self.partial_since = now
        if left is None and right is None:
            if self.missing_since is None:
                self.missing_since = now
            elapsed = now - self.missing_since
            self.status = f"Both boundaries missing: {elapsed:.1f}/{self.no_boundary_timeout:g}s"
            if elapsed >= self.no_boundary_timeout:
                return "both lane boundaries lost"
        else:
            self.missing_since = None
            elapsed = now - self.partial_since
            missing = "yellow left" if left is None else "white right"
            self.status = f"Recovering {missing} boundary: {elapsed:.1f}/{self.single_boundary_timeout:g}s"
        if now - self.partial_since >= self.single_boundary_timeout:
            return "lane boundaries not recovered"
        return None


def drive(model, bot_name, speed, duration, preview=True, gain=MOTOR_GAIN, camera_settings=None, output=None,
          no_boundary_timeout=.5, single_boundary_timeout=3.,
          crop_top=REAL_CROP_TOP, steering_gain=REAL_STEERING_GAIN):
    """Run a bounded physical trial; always stop motors and retain diagnostics."""
    guard = LaneVisibilityGuard(no_boundary_timeout, single_boundary_timeout)
    settings = real_camera_settings(model.camera_settings if camera_settings is None else camera_settings, crop_top)
    real_wheels(2, speed, gain, steering_gain)
    bot_name = bot_name.removesuffix(".local")
    robot = DB21J(bot_name, simulated=False)
    robot.add("motors", "base", RealMotors(bot_name))
    viewer = CameraPreview(window_title="Duckiebot real driving — Q stops motors") if preview else None
    detector = LineDetector()
    detector.camera_settings = settings
    history = CameraHistory()
    output = Path(output) if output is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    samples = []
    reason = "duration completed"
    started = None
    frame = None
    reported = -1.
    try:
        robot.motors.start()
        robot.motors.set_pwm(left=0., right=0.)
        robot.camera.start()
        frame = robot.camera.capture(block=True, timeout=10.)
        if frame is None:
            raise RuntimeError("No camera frame received from the Duckiebot.")
        started = last_frame = time.monotonic()
        if output is not None:
            cv2.imwrite(str(output / "camera_start.png"), frame)
        while time.monotonic() - started < duration:
            image = camera_input(frame[:, :, [2, 1, 0]], settings)
            observation = history.push(line_mask(image, settings))
            left_line, right_line = detector.detect(image)
            lost = guard.update(left_line, right_line)
            if not samples and (left_line is None or right_line is None):
                lost = "both boundaries must be visible before starting"
            detector._guidance_status = f"STOP: {lost}" if lost else guard.status
            if lost:
                # Stop before GUI drawing, which can take longer than a frame.
                robot.motors.set_pwm(left=0., right=0.)
                reason = lost
            if viewer is not None:
                viewer.show(image, detector, raw_image=frame[:, :, [2, 1, 0]], cnn_input=observation)
            if lost:
                break
            action, _ = model.predict(observation, deterministic=True)
            left, right = real_wheels(action, speed, gain, steering_gain)
            robot.motors.set_pwm(left=float(left), right=float(right))
            elapsed = time.monotonic() - started
            samples.append(dict(seconds=elapsed, action=ACTION_NAMES[int(action)],
                                left_pwm=float(left), right_pwm=float(right),
                                left_boundary=left_line is not None, right_boundary=right_line is not None,
                                paint_pixels=int((observation[:, :, -1] > 0).sum()),
                                guidance_status=guard.status))
            if elapsed - reported >= 1.:
                print(f"real seconds={elapsed:.1f} action={ACTION_NAMES[int(action)]} "
                      f"left={float(left):.3f} right={float(right):.3f} "
                      f"guidance={guard.status}", flush=True)
                reported = elapsed
            remaining = .05 - (time.monotonic() - last_frame)
            if remaining > 0:
                time.sleep(remaining)
            frame = robot.camera.capture(block=True, timeout=.5)
            last_frame = time.monotonic()
            if frame is None:
                raise RuntimeError("Camera timed out; stopping the Duckiebot.")
    except KeyboardInterrupt:
        reason = "stopped by user"
        raise
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            robot.motors.set_pwm(left=0., right=0.)
        finally:
            for component in (robot.camera, robot.motors):
                if component.is_started:
                    try:
                        component.stop()
                    except Exception as error:
                        print(f"Component shutdown: {error}")
            if viewer is not None:
                viewer.close()
            elapsed = time.monotonic() - started if started is not None else 0.
            report = dict(hostname=f"{bot_name}.local", requested_seconds=duration, seconds=elapsed,
                          steps=len(samples), stop_reason=reason, motors_stop_sent=True,
                          completed=reason == "duration completed" and bool(samples),
                          both_lines_fraction=sum(s["left_boundary"] and s["right_boundary"] for s in samples) / max(len(samples), 1),
                          speed=speed, motor_gain=gain, steering_gain=steering_gain,
                          camera_settings=settings, samples=samples,
                          no_boundary_timeout=no_boundary_timeout,
                          single_boundary_timeout=single_boundary_timeout)
            if output is not None:
                if frame is not None:
                    cv2.imwrite(str(output / "camera_end.png"), frame)
                    final_image = camera_input(frame[:, :, [2, 1, 0]], settings)
                    cv2.imwrite(str(output / "mask_end.png"), line_mask(final_image, settings))
                (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Real trial stopped: {reason}; steps={len(samples)}; motors stopped.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the DQN model .zip")
    parser.add_argument("duckiebot", help="Duckiebot name, e.g. duckie01")
    parser.add_argument("--speed", type=float,
                        help="override the total wheel-speed budget, bypassing MOTOR_GAIN")
    parser.add_argument("--crop-top", type=float, default=REAL_CROP_TOP,
                        help="fraction cropped from the top of the real camera (default: 0.4)")
    parser.add_argument("--steering-gain", type=float, default=REAL_STEERING_GAIN,
                        help="turn differential: 0.6 gives 2.125:1; 1 gives 4:1 (default: 1)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="driving time in seconds (default: 60)")
    parser.add_argument("--no-preview", action="store_true",
                        help="drive without the camera visualizer")
    parser.add_argument("--floor-filter", action=argparse.BooleanOptionalAction, default=None,
                        help="override the model's saved floor filter; preview and fine-tune before driving")
    parser.add_argument("--road-value-max", type=int, help="dark-road brightness limit (0–255)")
    parser.add_argument("--output", type=Path, help="save physical-trial diagnostics and camera snapshots")
    parser.add_argument("--single-boundary-timeout", type=float, default=5.,
                        help="seconds to recover both boundaries while at least one remains visible (default: 5)")
    parser.add_argument("--no-boundary-timeout", type=float, default=1,
                        help="seconds allowed with neither boundary visible (default: 1)")
    args = parser.parse_args()
    if args.road_value_max is not None and not 0 <= args.road_value_max <= 255:
        parser.error("road-value-max must be between 0 and 255")
    if not args.model.is_file() or args.model.suffix.lower() != ".zip":
        parser.error("model must be an existing .zip file")
    if not 0 < args.duration < float("inf"):
        parser.error("--duration must be positive and finite")
    try:
        LaneVisibilityGuard(args.no_boundary_timeout, args.single_boundary_timeout)
    except ValueError as error:
        parser.error(str(error))
    # Validate the model before connecting to the motors.
    model = DQN.load(args.model, device="auto")
    try:
        speed = saved_speed(model) if args.speed is None else args.speed
        validate_model(model)
        gain = MOTOR_GAIN if args.speed is None else 1.0
        # Validate commands before starting hardware; no model-saved gain
        # overrides the editable constant above.
        straight = real_wheels(2, speed, gain, args.steering_gain)
        settings = real_camera_settings(
            floor_camera_settings(model.camera_settings, args.floor_filter, args.road_value_max), args.crop_top)
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Driving {args.duckiebot} with {args.model}, speed={speed:g}, "
        f"motor_gain={gain:g}, straight PWM={float(straight[0]):g}, "
        f"crop_top={args.crop_top:g}, steering_gain={args.steering_gain:g}. Ctrl+C stops."
    )
    try:
        preview = not args.no_preview
        report = drive(model, args.duckiebot, speed, args.duration, preview=preview, gain=gain, camera_settings=settings, output=args.output,
                       no_boundary_timeout=args.no_boundary_timeout,
                       single_boundary_timeout=args.single_boundary_timeout,
                       crop_top=args.crop_top, steering_gain=args.steering_gain)
        return 0 if report["completed"] else 1
    except KeyboardInterrupt:
        print("Stopped by driver.")
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
