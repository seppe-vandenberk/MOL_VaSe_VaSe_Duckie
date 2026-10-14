"""Local Matplotlib preview of the camera, detections, and CNN frame stack."""

import time
import cv2
import numpy as np


def _preview_worker(frames, shutdown, stop_requested):
    """Keep GUI drawing off the training thread; show supplied observations."""
    from queue import Empty
    viewer = CameraPreview()
    first = True
    try:
        while not shutdown.is_set():
            try:
                observation, status, raw_image, cnn_input = frames.get(timeout=.05)
            except Empty:
                if viewer.figure is not None:
                    viewer.figure.canvas.flush_events()
                if viewer.stop_requested:
                    stop_requested.set()
                    break
                continue
            try:
                viewer.show(observation, status, raw_image=raw_image, cnn_input=cnn_input)
            except KeyboardInterrupt:
                stop_requested.set()
                break
            if not viewer.preview:
                stop_requested.set()
                raise RuntimeError("A desktop display is required for the CNN viewer")
            if first and viewer.figure is not None:
                print(f"CNN viewer opened: mask shape={cnn_input.shape}; paint=255, road=0.", flush=True)
                first = False
    finally:
        viewer.close()


class AsyncCameraPreview:
    """Publish exact policy masks to a separate GUI process at most 5 Hz."""
    def __init__(self):
        import multiprocessing
        context = multiprocessing.get_context("spawn")
        self.frames = context.Queue(maxsize=1)
        self.shutdown = context.Event()
        self.stop_requested = context.Event()
        self.process = context.Process(target=_preview_worker,
                                       args=(self.frames, self.shutdown, self.stop_requested),
                                       name="Duckiebot CNN viewer", daemon=True)
        self.preview = True
        self.started = False
        self.last_frame = 0.

    def show(self, observation, status=None, raw_image=None, cnn_input=None):
        from queue import Empty, Full
        from types import SimpleNamespace
        if self.stop_requested.is_set():
            raise KeyboardInterrupt
        if not self.started:
            self.process.start()
            self.started = True
        elif not self.process.is_alive():
            raise RuntimeError("Training viewer exited unexpectedly")
        now = time.monotonic()
        if now - self.last_frame < .2:
            return
        snapshot = SimpleNamespace(**{name: getattr(status, name, None) for name in
                                     ("_selected_mask", "_left_line", "_right_line",
                                      "_left_line_type", "_right_line_type")})
        if snapshot._selected_mask is not None:
            snapshot._selected_mask = snapshot._selected_mask.copy()
        payload = (observation.copy(), snapshot,
                   None if raw_image is None else raw_image.copy(),
                   None if cnn_input is None else cnn_input.copy())
        try:
            self.frames.put_nowait(payload)
        except Full:
            try:
                self.frames.get_nowait()
            except Empty:
                pass
            try:
                self.frames.put_nowait(payload)
            except Full:
                pass
        self.last_frame = now

    def close(self):
        self.shutdown.set()
        if self.started:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        self.frames.cancel_join_thread()
        self.frames.close()
        self.preview = False


class CameraPreview:
    def __init__(self, window_title="Duckiebot training — exact CNN input"):
        self.window_title = window_title
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
                self.figure.canvas.manager.set_window_title(self.window_title)
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
        guidance = getattr(status, "_guidance_status", None)
        if guidance:
            title += f"\n{guidance}"
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
