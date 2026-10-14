"""Detect the yellow left and white right boundaries in an RGB road crop."""

import cv2
import numpy as np
from utils.paint_mask import paint_masks


class LineDetector:
    minimum_paint_width = 2
    minimum_vertical_span_fraction = 0.12
    minimum_paint_brightness = 120
    minimum_dash_gap = 2
    edge_band_fraction = 0.35

    def detect(self, image):
        left, right = self._find_road_lines(image)
        self._left_line = None if left is None else left[2]
        self._right_line = None if right is None else right[2]
        self._left_line_type = None if left is None else left[4]
        self._right_line_type = None if right is None else right[4]
        return left, right

    def _find_road_lines(self, image):
        """Find thick bright painted regions, then fit their direction."""
        height, width = image.shape[:2]
        # The image is already cropped and resized to the policy input.
        roi_top = 0
        bottom = height - 1
        # HSV is only used to select paint, as in the original RGB detector.
        settings = dict(getattr(self, "camera_settings", {}))
        settings.setdefault("white_hsv_lower", (0, 0, self.minimum_paint_brightness))
        white, yellow = paint_masks(image, settings)
        paint = cv2.bitwise_or(white, yellow)
        paint[:roi_top, :] = 0

        # Opening removes markings too thin to contain this round brush.
        # Unlike edge detection, it measures the filled paint's thickness.
        brush_width = self.minimum_paint_width
        brush = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (brush_width, brush_width)
        )
        thick = cv2.morphologyEx(paint, cv2.MORPH_OPEN, brush)
        # Keep thin extensions if they connect to a thick painted section.
        _, labels = cv2.connectedComponents(paint)
        accepted = np.unique(labels[thick > 0])
        accepted = accepted[accepted != 0]
        self._paint_mask = (np.isin(labels, accepted) * 255).astype(np.uint8)
        self._selected_mask = np.zeros_like(paint)
        contours, _ = cv2.findContours(
            self._paint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Fit each paint piece first. Dashes are separate contours, so group
        # pieces that project onto the same path before choosing two lines.
        groups = []
        # Work from near to far, linking short dots as well as long dashes.
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1], reverse=True)
        for contour in contours:
            x, y, box_width, span = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 1:
                continue
            dx, dy, cx, cy = cv2.fitLine(
                contour, cv2.DIST_L2, 0, 0.01, 0.01
            ).reshape(4)
            dot_like = span <= 5
            if abs(dy) < 0.01 and not dot_like:
                continue
            if dot_like:
                dx, dy = 0.0, 1.0
            bottom_x = float(cx + (bottom - cy) * dx / dy)
            slope = float(dx / dy)
            for group in groups:
                aligned = (
                    abs(bottom_x - group["bottom_x"]) < width * 0.08
                    and abs(slope - group["slope"]) < 0.25
                )
                # Rounded dashes on a bend have unreliable individual angles.
                # Link nearby pieces using their visible centres instead.
                nearby_dot = (
                    (dot_like or group["dot_like"])
                    and abs(float(cx) - group["cx"]) < width * 0.15
                    and abs(float(cy) - group["cy"]) < height * 0.28
                )
                if aligned or nearby_dot:
                    group["pieces"].append(contour)
                    group["cx"], group["cy"] = float(cx), float(cy)
                    group["dot_like"] = group["dot_like"] or dot_like
                    break
            else:
                groups.append({
                    "bottom_x": bottom_x,
                    "slope": slope,
                    "pieces": [contour],
                    "cx": float(cx), "cy": float(cy),
                    "dot_like": dot_like,
                })

        left_candidates = []
        right_candidates = []
        for group in groups:
            pieces = group["pieces"]
            points = np.concatenate(pieces)
            _, y, _, span = cv2.boundingRect(points)
            if (
                span < height * self.minimum_vertical_span_fraction
                or y + span < height * 0.50
            ):
                continue
            dx, dy, cx, cy = cv2.fitLine(
                points, cv2.DIST_L2, 0, 0.01, 0.01
            ).reshape(4)
            if abs(dy) < 0.01:
                continue
            bottom_x = float(cx + (bottom - cy) * dx / dy)
            # Use the visible lower endpoint for alignment; extrapolation to
            # the bottom edge can be wildly wrong on a curved marking.
            lower_y = min(bottom, y + span - 1)
            lower_points = points.reshape(-1, 2)
            lower_points = lower_points[lower_points[:, 1] >= lower_y - 2]
            visible_x = float(np.median(lower_points[:, 0]))
            # Alignment uses the visible endpoint as well.
            bottom_x = visible_x
            line = (
                int(visible_x), lower_y,
                int(cx + (y - cy) * dx / dy), y,
            )
            # Gaps between aligned pieces distinguish dashed from solid paint.
            intervals = sorted(
                (cv2.boundingRect(piece)[1],
                 cv2.boundingRect(piece)[1] + cv2.boundingRect(piece)[3])
                for piece in pieces
            )
            covered_end = intervals[0][1]
            gaps = 0
            for start, end in intervals[1:]:
                if start - covered_end >= self.minimum_dash_gap:
                    gaps += 1
                covered_end = max(covered_end, end)
            kind = "dashed" if gaps else "solid"
            area = sum(cv2.contourArea(piece) for piece in pieces)
            candidate = (area, bottom_x, line, pieces, kind)
            # Paint identity survives a boundary crossing the image centre.
            # Position alone swaps left/right precisely when recovery is needed.
            xs, ys = points.reshape(-1, 2).T
            is_yellow = np.count_nonzero(yellow[ys, xs]) > np.count_nonzero(white[ys, xs])
            if is_yellow:
                left_candidates.append(candidate)
            else:
                right_candidates.append(candidate)

        # The desired lane has dashed paint on the left and solid on the right.
        pairs = [
            (left, right)
            for left in left_candidates
            for right in right_candidates
            if left[4] == "dashed" and right[4] == "solid"
        ]
        if pairs:
            left, right = max(pairs, key=lambda pair: pair[0][0] + pair[1][0])
        else:
            left = max(left_candidates, default=None, key=lambda item: item[0])
            right = max(right_candidates, default=None, key=lambda item: item[0])
        for candidate in (left, right):
            if candidate is not None:
                cv2.drawContours(self._selected_mask, candidate[3], -1, 255, -1)
        return left, right
