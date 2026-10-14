"""Preview a real Duckiebot's normal camera view without driving."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time
from stable_baselines3 import DQN

from duckietown.sdk.robots.duckiebot import DB21J

from training.environment import camera_input, line_mask, CameraHistory, CAMERA_SETTINGS
from utils.utils import validate_model
from utils.paint_mask import floor_camera_settings
from utils.real_settings import REAL_CROP_TOP, real_camera_settings
from scripts.visualizer import CameraPreview
from utils.line_detection import LineDetector


# Set this to the real bot's camera service port, or pass --port.
CAMERA_PORT = 11911


def show_camera(bot_name, settings=None, *, simulated=False, port=CAMERA_PORT, crop_top=REAL_CROP_TOP):
    bot_name = bot_name if simulated else bot_name.removesuffix(".local")
    title = f"Duckiematrix camera — port {port}" if simulated else "Duckiebot real camera — camera only"
    viewer = CameraPreview(window_title=title)
    detector = LineDetector()
    settings = CAMERA_SETTINGS if settings is None else settings
    if not simulated:
        settings = real_camera_settings(settings, crop_top)
    detector.camera_settings = settings
    history = CameraHistory()
    robot = DB21J(bot_name, simulated=simulated, port=port)
    try:
        robot.camera.start()
        print(f"Viewing {bot_name}; Q or Ctrl+C stops. Motors are not started.")
        last_frame = 0.0
        first_frame = True
        while True:
            remaining = 0.05 - (time.monotonic() - last_frame)
            if remaining > 0:
                time.sleep(remaining)
            frame = robot.camera.capture(
                block=True, timeout=10.0 if first_frame else 2.0
            )
            if frame is None:
                raise RuntimeError("Camera timed out.")
            first_frame = False
            last_frame = time.monotonic()
            image = camera_input(frame[:, :, [2, 1, 0]], settings)
            detector.detect(image)
            viewer.show(image, detector, raw_image=frame[:, :, [2,1,0]],
                        cnn_input=history.push(line_mask(image, settings)))
            if not viewer.preview:
                raise RuntimeError("A desktop display is needed for the Matplotlib preview.")
    finally:
        try:
            if robot.camera.is_started:
                robot.camera.stop()
        finally:
            viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("duckiebot", help="bot name, e.g. quackenstein")
    parser.add_argument("--port", type=int, default=CAMERA_PORT, help="camera service port")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/baseline.zip",
                        help="use the exact camera preprocessing saved in this model zip")
    parser.add_argument("--floor-filter", action=argparse.BooleanOptionalAction, default=None,
                        help="reject broad pale floor regions in the displayed CNN mask")
    parser.add_argument("--road-value-max", type=int, help="dark-road brightness limit (0–255)")
    parser.add_argument("--crop-top", type=float, default=REAL_CROP_TOP,
                        help="fraction cropped from the top, matching run_real.py (default: 0.4)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.road_value_max is not None and not 0 <= args.road_value_max <= 255:
        parser.error("road-value-max must be between 0 and 255")
    settings = None
    if args.model:
        if not args.model.is_file():
            parser.error("--model must be an existing model zip")
        model = DQN.load(args.model, device="cpu")
        try:
            validate_model(model)
        except ValueError as error:
            parser.error(str(error))
        settings = model.camera_settings
    if args.floor_filter is not None or args.road_value_max is not None:
        settings = floor_camera_settings(CAMERA_SETTINGS if settings is None else settings,
                                         args.floor_filter, args.road_value_max)
    try:
        settings = real_camera_settings(CAMERA_SETTINGS if settings is None else settings, args.crop_top)
        show_camera(args.duckiebot, settings=settings, port=args.port, crop_top=args.crop_top)
    except KeyboardInterrupt:
        print("Camera preview stopped.")
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
