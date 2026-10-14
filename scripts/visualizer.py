"""Local Matplotlib preview of the camera, detections, and CNN frame stack."""

import time
import cv2
import numpy as np


class CameraPreview:
    def __init__(self):
        self.preview = True
        self.figure = self.plot = self.detection_plot = None
        self.cnn_plots = []
        self.stop_requested = False
        self.last_frame = 0.

    def show(self, observation, status=None, raw_image=None, cnn_input=None):
        if not self.preview:
            return
        import matplotlib.pyplot as plt
        if self.stop_requested:
            raise KeyboardInterrupt
        now = time.monotonic()
        if self.figure is not None and now - self.last_frame < .2:
            self.figure.canvas.flush_events()
            if self.stop_requested:
                raise KeyboardInterrupt
            return
        raw = observation if raw_image is None else raw_image
        # The policy transposes HWC to CHW and divides these same pixels by 255.
        frames = ([observation] if cnn_input is None else
                  [cnn_input[:, :, index] for index in range(cnn_input.shape[2])])
        annotated = observation.copy()
        mask = getattr(status,"_selected_mask",None)
        if mask is not None:
            selected = mask > 0
            annotated[selected] = (annotated[selected] * .5 + (0,127.5,0)).astype(np.uint8)
        for side in ("left","right"):
            line = getattr(status,f"_{side}_line",None)
            if line is not None:
                x1,y1,x2,y2 = line
                cv2.line(annotated,(x1,y1),(x2,y2),(0,255,0),1)
        if self.figure is None:
            try:
                if plt.get_backend().lower() == "agg":
                    plt.switch_backend("TkAgg")
                self.figure = plt.figure(figsize=(12,8), layout="constrained")
                grid = self.figure.add_gridspec(2, 2 * len(frames))
                axes = [self.figure.add_subplot(grid[0, :len(frames)]),
                        self.figure.add_subplot(grid[0, len(frames):])]
                axes[0].set_title("Raw RGB camera")
                for axis in axes:
                    axis.axis("off")
                self.plot = axes[0].imshow(raw, interpolation="nearest")
                self.detection_plot = axes[1].imshow(annotated, interpolation="nearest")
                for index, frame in enumerate(frames):
                    axis = self.figure.add_subplot(grid[1, 2 * index:2 * index + 2])
                    age = len(frames) - index - 1
                    label = "newest" if age == 0 else f"t−{age}"
                    axis.set_title(f"CNN line mask: {label}" if cnn_input is not None
                                   else "RGB crop")
                    axis.axis("off")
                    self.cnn_plots.append(axis.imshow(frame, interpolation="nearest",
                                                     cmap="gray", vmin=0, vmax=255))
                self.figure.canvas.mpl_connect("key_press_event", self._preview_key)
                self.figure.canvas.mpl_connect("close_event", self._preview_closed)
                plt.show(block=False)
            except (ImportError, RuntimeError) as error:
                print(f"Matplotlib preview unavailable: {error}")
                self.close()
                self.preview = False
                return
        self.plot.set_data(raw)
        self.detection_plot.set_data(annotated)
        for plot, frame in zip(self.cnn_plots, frames):
            plot.set_data(frame)
        height, width = observation.shape[:2]
        title = f"{width}×{height} RGB crop with detections (Q to stop)"
        title += f"\nLeft: {getattr(status,'_left_line_type',None) or 'missing'}; right: {getattr(status,'_right_line_type',None) or 'missing'}"
        self.detection_plot.axes.set_title(title)
        self.figure.suptitle(
            f"CNN input: {len(frames) if cnn_input is not None else 3}×{height}×{width} "
            "(channels×height×width); pixels normalized to [0, 1]"
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.last_frame = now
        if self.stop_requested:
            raise KeyboardInterrupt

    def _preview_key(self, event):
        if event.key and event.key.lower() == "q":
            self.stop_requested = True

    def _preview_closed(self, event):
        self.stop_requested = True

    def close(self):
        if self.figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.figure)
            self.figure = None
            self.cnn_plots = []
