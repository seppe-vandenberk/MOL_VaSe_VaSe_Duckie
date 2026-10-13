"""Duckiematrix adaptation of Almasi et al.'s lane geometry reward.

R = 10 * speed * heading_alignment - 100 * abs(lane_distance) in the right lane.
Wrong lane: zero. Outside the road: -40 and terminate.
The SDK has no collision-penalty signal; this obstacle-free adaptation omits it.
"""

import math
import time
import gymnasium as gym
import numpy as np
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.utils.lane_position import MapInterpreter, bezier_closest, bezier_point, bezier_tangent


class LaneGeometry:
    """Directed right-lane curves, with continuous checks across tile seams."""

    def __init__(self, map_data, namespace):
        frames = map_data["frames"]["data"]
        # Some SDK maps advertise tiles from other maps without frame poses.
        tiles = {name: tile for name, tile in map_data["tiles"]["data"].items()
                 if name.startswith(namespace + "/tile_") and name in frames}
        data = dict(map_data)
        data["tiles"] = {"data": tiles}
        # The installed SDK hard-codes this tile_info key, even for other maps.
        data["tile_info"] = {"data": {"map_0": map_data["tile_info"]["data"][namespace]}}
        interpreter = MapInterpreter(data)
        self.tile_size = interpreter.road_tile_size
        self.curves = []
        for tile in interpreter.drivable_tiles:
            centre = np.array([tile["pose"]["x"], tile["pose"]["y"]]) * self.tile_size
            curves = tile["curves"]
            midpoints = [bezier_point(curve, .5)[[0,2]] for curve in curves]
            road_middle = np.mean(midpoints, axis=0)
            for curve, midpoint in zip(curves, midpoints):
                tangent = bezier_tangent(curve, .5)[[0,2]]
                right = np.array([tangent[1], -tangent[0]])
                # Ensure right-lane orientation in the ground X/Y coordinate plane.
                if len(curves) == 2 and np.dot(midpoint - road_middle, right) < 0:
                    curve = curve[::-1].copy()
                self.curves.append((centre, curve))
        if not self.curves:
            raise RuntimeError("Map has no drivable lane curves")

    def measure(self, position, heading):
        point3 = np.array([position[0], 0., position[1]])
        candidates = []
        for centre, curve in self.curves:
            if np.max(np.abs(position - centre)) > self.tile_size * 1.5:
                continue
            t = bezier_closest(curve, point3)
            point = bezier_point(curve, t)[[0,2]]
            tangent = bezier_tangent(curve, t)[[0,2]]
            length = np.linalg.norm(tangent)
            if length <= 1e-9:
                continue
            tangent /= length
            distance = float(np.linalg.norm(position - point))
            alignment = float(np.clip(np.dot(heading, tangent), -1., 1.))
            candidates.append((distance, alignment))
        # Lane centres lie 0.20 tiles from road centre; edge is at 0.50.
        # Two percent tolerance avoids false failures at approximate tile seams.
        if not candidates or min(item[0] for item in candidates) > .32 * self.tile_size:
            return {"off_road": True, "in_right_lane": False, "lane_distance": 0., "heading_alignment": 0.}
        matching = [item for item in candidates if item[1] > 0]
        distance, alignment = min(matching or candidates, key=lambda item: item[0])
        return {"off_road": False, "in_right_lane": distance <= .25 * self.tile_size and alignment > 0,
                "lane_distance": distance, "heading_alignment": alignment}


class LaneReward(gym.Env):
    REWARD_VERSION = "paper_geometry_duckiematrix_v1"
    observation_interval = .05
    speed_gain = 10.
    distance_penalty = 100.
    off_road_penalty = 40.
    max_episode_steps = 2000

    def __init__(self, entity_name="map_0/vehicle_0", *, camera_settings=None):
        super().__init__()
        from training.environment import CAMERA_SETTINGS
        self.camera_settings = dict(CAMERA_SETTINGS if camera_settings is None else camera_settings)
        self.robot = DB21J(entity_name, simulated=True)
        self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
        self.observation_space = gym.spaces.Box(0, 255, (480,640,3), np.uint8)
        self._last_frame_time = None
        self._components = (self.robot.camera, self.robot.motors, self.robot.reset_flag, self.robot.pose)
        try:
            for component in self._components:
                component.start()
            map_components = {"frames": self.robot.map_frames, "tiles": self.robot.map_tiles,
                              "tile_info": self.robot.map_tile_info}
            try:
                for component in map_components.values():
                    component.start()
                data = {name: component.capture(block=True, timeout=5)
                        for name, component in map_components.items()}
                if any(value is None for value in data.values()):
                    raise RuntimeError("Simulator map unavailable")
                self.geometry = LaneGeometry(data, entity_name.split("/")[0])
            finally:
                for component in map_components.values():
                    if component.is_started:
                        component.stop()
        except Exception:
            self.close()
            raise

    def reward_settings(self):
        return {name: getattr(self, name) for name in (
            "observation_interval", "speed_gain", "distance_penalty", "off_road_penalty", "max_episode_steps")}

    def _camera(self):
        if self._last_frame_time is not None:
            remaining = self.observation_interval - (time.monotonic() - self._last_frame_time)
            if remaining > 0:
                time.sleep(remaining)
        bgr = self.robot.camera.capture(block=True, timeout=2)
        if bgr is None:
            raise RuntimeError("Simulator camera timed out")
        self._last_frame_time = time.monotonic()
        return bgr[:, :, [2,1,0]]

    def _pose(self):
        pose = self.robot.pose.capture(block=True, timeout=2)
        if pose is None:
            raise RuntimeError("Simulator pose unavailable")
        position = np.array([pose["position"]["x"], pose["position"]["y"]], dtype=float)
        q = pose["rotation"]
        yaw = math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]),
                         1 - 2 * (q["y"]**2 + q["z"]**2))
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        stamp = float(pose["header"]["timestamp"])
        if not np.all(np.isfinite(position)) or not np.isfinite(stamp) or not np.all(np.isfinite(heading)):
            raise RuntimeError("Simulator returned an invalid pose")
        return position, heading, stamp

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.robot.motors.set_pwm(left=0., right=0.)
        self.robot.reset_flag.set_reset(True)
        image = self._camera()
        self._last_position, heading, self._last_stamp = self._pose()
        self._lane = self.geometry.measure(self._last_position, heading)
        self._speed = self._step_distance = self._distance_driven = self._right_lane_distance = 0.
        self._episode_steps = 0
        self._termination_reason = None
        return image, self._get_info()

    def reward_fn(self):
        if self._lane["off_road"]:
            self._termination_reason = "outside geometric road boundary"
            return -self.off_road_penalty, True
        if not self._lane["in_right_lane"]:
            return 0., False
        reward = self.speed_gain * self._speed * self._lane["heading_alignment"]
        reward -= self.distance_penalty * self._lane["lane_distance"]
        return float(reward), False

    def step(self, actions):
        left, right = np.asarray(actions, dtype=float).reshape(2)
        self.robot.motors.set_pwm(left=float(left), right=float(right))
        try:
            image = self._camera()
            position, heading, stamp = self._pose()
            distance = float(np.linalg.norm(position - self._last_position))
            dt = stamp - self._last_stamp
            # Ignore teleport/reset jumps; duplicate timestamps give zero speed.
            self._step_distance = distance if distance <= .5 and dt > 0 else 0.
            self._speed = self._step_distance / dt if dt > 0 else 0.
            self._last_position, self._last_stamp = position, stamp
            self._distance_driven += self._step_distance
            self._lane = self.geometry.measure(position, heading)
            if self._lane["in_right_lane"] and not self._lane["off_road"]:
                self._right_lane_distance += self._step_distance
            self._episode_steps += 1
            self._termination_reason = None
            reward, terminated = self.reward_fn()
            truncated = self._episode_steps >= self.max_episode_steps and not terminated
            if truncated:
                self._termination_reason = "episode time limit"
            if terminated or truncated:
                self.robot.motors.set_pwm(left=0., right=0.)
            return image, reward, terminated, truncated, self._get_info()
        except BaseException:
            self.robot.motors.set_pwm(left=0., right=0.)
            raise

    def _get_info(self):
        return {**self._lane, "speed_m_s": self._speed, "step_distance_m": self._step_distance,
                "distance_driven_m": self._distance_driven, "right_lane_distance_m": self._right_lane_distance,
                "collision_penalty": 0., "termination_reason": self._termination_reason}

    def close(self):
        try:
            self.robot.motors.set_pwm(left=0., right=0.)
        finally:
            for component in self._components:
                if component.is_started:
                    component.stop()
