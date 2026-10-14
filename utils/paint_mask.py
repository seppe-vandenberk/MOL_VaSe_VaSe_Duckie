"""Shared paint selection and optional rejection of purple carpet or bright floor."""

import cv2
import numpy as np

DEFAULT_FLOOR_FILTER = {
    "enabled": False,
    "road_value_max": 110,
    "road_saturation_max": 120,
    "minimum_road_fraction": .02,
    "maximum_paint_area_fraction": .35,
    "maximum_paint_width_fraction": .25,
    "minimum_tape_aspect_ratio": 2.5,
    "road_neighbourhood": 2,
    "excluded_floor_hsv_lower": [130, 20, 0],
    "excluded_floor_hsv_upper": [179, 255, 255],
}


def floor_camera_settings(settings, enabled=None, road_value_max=None):
    """Copy settings so preview overrides never mutate a loaded checkpoint."""
    copied = dict(settings)
    floor = dict(DEFAULT_FLOOR_FILTER)
    floor.update(copied.get("floor_filter", {}))
    if enabled is not None:
        floor["enabled"] = bool(enabled)
    if road_value_max is not None:
        if not 0 <= road_value_max <= 255:
            raise ValueError("Road brightness must be between 0 and 255")
        floor["road_value_max"] = int(road_value_max)
    copied["floor_filter"] = floor
    return copied


class FloorPaintFilter:
    """Keep compact paint pieces adjacent to the main dark road component.

    Assumes white/yellow markings on a dark road or mat. Purple carpet is
    excluded by hue; broad bright floor components are rejected by thickness.
    This is an image heuristic; its thresholds need checking on the real camera.
    """

    def __init__(self, settings=None):
        self.settings = dict(DEFAULT_FLOOR_FILTER)
        self.settings.update(settings or {})

    def apply(self, hsv, white, yellow):
        settings = self.settings
        if not settings["enabled"]:
            return white, yellow
        # Washed-out purple carpet may satisfy the white threshold; dark
        # purple carpet may satisfy the road threshold. Exclude it from both.
        excluded = cv2.inRange(hsv, tuple(settings["excluded_floor_hsv_lower"]),
                               tuple(settings["excluded_floor_hsv_upper"]))
        keep = cv2.bitwise_not(excluded)
        white = cv2.bitwise_and(white, keep)
        yellow = cv2.bitwise_and(yellow, keep)
        road = cv2.inRange(hsv, (0, 0, 0),
                           (179, int(settings["road_saturation_max"]), int(settings["road_value_max"])))
        road = cv2.bitwise_and(road, keep)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(road)
        empty = np.zeros_like(white)
        if count <= 1:
            return empty, empty.copy()
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[main, cv2.CC_STAT_AREA] < road.size * settings["minimum_road_fraction"]:
            return empty, empty.copy()
        road = np.uint8(labels == main) * 255
        radius = max(0, int(settings["road_neighbourhood"]))
        road = cv2.dilate(road, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        return self._supported_paint(white, road), self._supported_paint(yellow, road)

    def _supported_paint(self, paint, road):
        height, width = paint.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(paint)
        output = np.zeros_like(paint)
        maximum_area = paint.size * self.settings["maximum_paint_area_fraction"]
        maximum_width = max(2., width * self.settings["maximum_paint_width_fraction"])
        for label in range(1, count):
            component = labels == label
            coordinates = np.float32(np.column_stack(np.nonzero(component)))
            _, dimensions, _ = cv2.minAreaRect(coordinates)
            aspect_ratio = max(dimensions) / max(min(dimensions), 1.)
            tape_shaped = aspect_ratio >= self.settings["minimum_tape_aspect_ratio"]
            # Distance to the component edge measures perpendicular thickness;
            # a nearby diagonal tape line can occupy many columns in each row.
            padded = np.pad(np.uint8(component), 1)
            distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
            thickness = 2. * float(distance.max()) - 1.
            # Nearby tape widens with perspective. Preserve an elongated
            # stroke even when its nearest section exceeds the width limit;
            # broad patches still fail the area/thickness checks.
            broad = thickness > maximum_width or stats[label, cv2.CC_STAT_AREA] > maximum_area
            if broad and not tape_shaped:
                continue
            if np.any(component & (road > 0)):
                output[component] = 255
        return output


def paint_masks(image, settings=None):
    """Return white/yellow masks used by both the reward and CNN input."""
    settings = {} if settings is None else settings
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    white = cv2.inRange(hsv, tuple(settings.get("white_hsv_lower", (0, 0, 120))),
                        tuple(settings.get("white_hsv_upper", (179, 100, 255))))
    yellow = cv2.inRange(hsv, tuple(settings.get("yellow_hsv_lower", (12, 60, 80))),
                         tuple(settings.get("yellow_hsv_upper", (42, 255, 255))))
    return FloorPaintFilter(settings.get("floor_filter")).apply(hsv, white, yellow)
