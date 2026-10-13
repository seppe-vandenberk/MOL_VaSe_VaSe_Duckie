"""Local Matplotlib preview of raw RGB and the paper's segmented CNN frame."""

import time


class CameraPreview:
    def __init__(self):
        self.preview = True
        self.figure = self.plot = self.detection_plot = None
        self.stop_requested = False
        self.last_frame = 0.

    def show(self, observation, status=None, raw_image=None):
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
        if self.figure is None:
            try:
                if plt.get_backend().lower() == "agg":
                    plt.switch_backend("TkAgg")
                self.figure, axes = plt.subplots(1,2,figsize=(11,5))
                axes[0].set_title("Raw RGB camera")
                for axis in axes:
                    axis.axis("off")
                self.plot = axes[0].imshow(raw, interpolation="nearest")
                self.detection_plot = axes[1].imshow(observation, interpolation="nearest")
                self.figure.canvas.mpl_connect("key_press_event", self._preview_key)
                self.figure.canvas.mpl_connect("close_event", self._preview_closed)
                plt.show(block=False)
            except (ImportError, RuntimeError) as error:
                print(f"Matplotlib preview unavailable: {error}")
                self.close()
                self.preview = False
                return
        self.plot.set_data(raw)
        self.detection_plot.set_data(observation)
        title = "CNN: five 40x80 frames (newest shown)\nYellow -> red; white -> green (Q to stop)"
        lane = getattr(status, "_lane", None)
        if lane is not None:
            title += f"\nLane distance={lane['lane_distance']:.3f} m; alignment={lane['heading_alignment']:.2f}"
        self.detection_plot.axes.set_title(title)
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
