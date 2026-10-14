"""Inspect a simulator camera and CNN masks without driving or resetting it."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

# Set this to 7501 + the offset printed by the training launcher.
SIMULATOR_PORT = 7501
ENTITY = "map_0/vehicle_0"

import argparse
from stable_baselines3 import DQN
from scripts.view_real import show_camera
from scripts.wait_matrix import wait_for_camera
from utils.paint_mask import floor_camera_settings
from utils.utils import validate_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=SIMULATOR_PORT)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/baseline.zip")
    parser.add_argument("--floor-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--road-value-max", type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        model = DQN.load(args.model, device="cpu")
        validate_model(model)
        settings = floor_camera_settings(model.camera_settings, args.floor_filter, args.road_value_max)
        wait_for_camera(args.entity, args.port, 10.)
        show_camera(args.entity, settings=settings, simulated=True, port=args.port)
    except KeyboardInterrupt:
        print("Simulator camera preview stopped.")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
