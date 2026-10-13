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

from training.environment import camera_input
from utils.utils import validate_model
from scripts.visualizer import CameraPreview


def show_camera(bot_name, settings=None):
    bot_name = bot_name.removesuffix(".local")
    viewer = CameraPreview()
    robot = DB21J(bot_name, simulated=False)
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
            viewer.show(image, raw_image=frame[:, :, [2,1,0]])
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
    parser.add_argument("--model", type=Path,
                        help="use the exact camera preprocessing saved in this model zip")
    args = parser.parse_args()
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
    try:
        show_camera(args.duckiebot, settings=settings)
    except KeyboardInterrupt:
        print("Camera preview stopped.")
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
