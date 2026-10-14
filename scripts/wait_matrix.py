"""Wait for a simulator camera frame without sending motor or reset commands."""

import argparse
import time
from urllib.request import Request, urlopen
from urllib.parse import quote
from urllib.error import URLError
from duckietown.sdk.robots.duckiebot import DB21J


def wait_for_camera(entity, port, timeout):
    deadline = time.monotonic() + timeout
    topic = f"http://127.0.0.1:{port}/robot/{quote(entity, safe='/')}/sensor/camera/front_center/jpeg/"
    print(f"Waiting for camera topic: {topic}", flush=True)
    while time.monotonic() < deadline:
        try:
            # The engine root responds before the renderer creates camera
            # topics. A premature SDK subscription fails once and never retries.
            with urlopen(Request(topic, method="HEAD"), timeout=1.):
                break
        except (URLError, TimeoutError):
            time.sleep(.2)
    else:
        raise RuntimeError(f"Simulator camera topic did not become ready before the startup timeout: {topic}")
    robot = DB21J(entity, simulated=True, port=port)
    try:
        robot.camera.start()
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            frame = robot.camera.capture(block=True, timeout=min(2., max(.01, remaining)))
            if frame is not None:
                print(f"Simulator camera ready: {entity}, port={port}", flush=True)
                return
        raise RuntimeError("Simulator camera did not become ready before the startup timeout")
    finally:
        if robot.camera.is_started:
            robot.camera.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="map_0/vehicle_0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=120.)
    args = parser.parse_args()
    wait_for_camera(args.entity, args.port, args.timeout)
