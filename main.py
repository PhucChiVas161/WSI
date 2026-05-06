from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from stitching_engine import ORBStitchingEngine, StitchResult
from tile_manager import SparseTileGrid
from vision_utils import CalibrationResult, crop_roi, detect_fov_circle, draw_calibration_overlay


@dataclass
class SharedState:
    calibration: Optional[CalibrationResult] = None
    tracking_result: Optional[StitchResult] = None
    tracking_lost: bool = False
    last_frame_time: float = 0.0
    reset_requested: bool = False


class VideoCaptureSource:
    def __init__(self, camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class ToupcamSource:
    def __init__(self) -> None:
        self.hcam = None
        self.buf = None
        self.frame_buffer: Optional[np.ndarray] = None
        self.width = 0
        self.height = 0
        self.available = False
        self._init_camera()

    def _init_camera(self) -> None:
        try:
            # Add the SDK DLL directory to the path
            sdk_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dll_path = os.path.join(sdk_root, "win", "x64")
            
            if not os.path.exists(dll_path):
                print(f"[Toupcam] DLL path not found: {dll_path}")
                return
            
            os.add_dll_directory(dll_path)
            sys.path.insert(0, sdk_root)

            import toupcam 

            cameras = toupcam.Toupcam.EnumV2()
            if not cameras:
                return

            print(f"[Toupcam] Found {len(cameras)} camera(s)")
            camera_info = cameras[0]
            print(f"[Toupcam] Opening: {camera_info.displayname}")

            self.hcam = toupcam.Toupcam.Open(camera_info.id)
            if not self.hcam:
                print("[Toupcam] Failed to open camera")
                return

            self.width, self.height = self.hcam.get_Size()
            bufsize = self._calc_bufsize()
            self.buf = bytes(bufsize)
            self.available = True
            print(f"[Toupcam] Initialized: {self.width}x{self.height}")
        except Exception as e:
            print(f"[Toupcam] Init failed: {e}")
            self.available = False

    def _calc_bufsize(self) -> int:
        try:
            import toupcam  # type: ignore

            return toupcam.TDIBWIDTHBYTES(self.width * 24) * self.height
        except Exception:
            return self.width * self.height * 3

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.available or self.hcam is None or self.buf is None:
            return False, None

        try:
            self.hcam.PullImageV4(self.buf, 0, 24, 0, None)
            frame = np.frombuffer(self.buf, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
            return True, frame
        except Exception as e:
            print(f"[Toupcam] Frame pull failed: {e}")
            return False, None

    def release(self) -> None:
        if self.hcam is not None:
            try:
                self.hcam.Close()
            except Exception:
                pass
            self.hcam = None
        self.buf = None


class FrameAcquisitionThread(threading.Thread):
    def __init__(self, source: VideoCaptureSource, frame_queue: queue.Queue, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            ok, frame = self.source.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
        self.source.release()


class ProcessingThread(threading.Thread):
    def __init__(
        self,
        frame_queue: queue.Queue,
        state: SharedState,
        stitcher: ORBStitchingEngine,
        tile_grid: SparseTileGrid,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.state = state
        self.stitcher = stitcher
        self.tile_grid = tile_grid
        self.stop_event = stop_event
        self.previous_roi: Optional[np.ndarray] = None
        self.world_from_frame = np.eye(3, dtype=np.float64)

    def _reset_tracking(self) -> None:
        self.previous_roi = None
        self.world_from_frame = np.eye(3, dtype=np.float64)
        self.state.tracking_result = None
        self.state.tracking_lost = False
        self.state.reset_requested = False

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self.state.last_frame_time = time.time()
            if self.state.reset_requested:
                self._reset_tracking()

            if self.state.calibration is None:
                self.state.calibration = detect_fov_circle(frame)

            calibration = self.state.calibration
            preview = draw_calibration_overlay(frame, calibration)
            roi = crop_roi(preview, calibration)
            roi_raw = crop_roi(frame, calibration)

            if self.previous_roi is None:
                self.previous_roi = roi_raw
                self.tile_grid.insert_frame(roi_raw, self.world_from_frame)
                self.state.tracking_result = StitchResult(True, np.eye(3, dtype=np.float64), reason="bootstrap")
                self.state.tracking_lost = False
                continue

            result = self.stitcher.estimate_motion(self.previous_roi, roi_raw)
            self.state.tracking_result = result
            self.state.tracking_lost = not result.success or result.confidence < 0.35

            if result.success:
                self.world_from_frame = self.world_from_frame @ np.linalg.inv(result.homography)
                self.tile_grid.insert_frame(roi_raw, self.world_from_frame)
                self.previous_roi = roi_raw
            elif self.state.tracking_lost:
                self.previous_roi = roi_raw


class App:
    def __init__(self, camera_index: int, tile_size: int, cache_limit: int, feather_px: int) -> None:
        self.stop_event = threading.Event()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.state = SharedState()
        self.source = self._select_camera_source(camera_index)
        self.stitcher = ORBStitchingEngine(matcher_type="bf")
        self.tile_grid = SparseTileGrid(tile_size=tile_size, cache_limit=cache_limit, feather_px=feather_px)
        self.acquisition_thread = FrameAcquisitionThread(self.source, self.frame_queue, self.stop_event)
        self.processing_thread = ProcessingThread(
            self.frame_queue, self.state, self.stitcher, self.tile_grid, self.stop_event
        )

    @staticmethod
    def _select_camera_source(camera_index: int):
        toupcam_source = ToupcamSource()
        if toupcam_source.available:
            return toupcam_source
        print("[Camera] No Toupcam detected, falling back to OpenCV")
        return VideoCaptureSource(camera_index=camera_index)

    def start(self) -> None:
        self.acquisition_thread.start()
        self.processing_thread.start()
        self._render_loop()

    def _render_loop(self) -> None:
        while not self.stop_event.is_set():
            frame = None
            if not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.queue[-1].copy()
                except Exception:
                    frame = None

            if frame is None:
                time.sleep(0.01)
                continue

            if self.state.calibration is not None:
                frame = draw_calibration_overlay(frame, self.state.calibration)

            if self.state.tracking_result is not None:
                result = self.state.tracking_result
                text = f"matches={result.matches} inliers={result.inliers} dx={result.dx:.1f} dy={result.dy:.1f} rot={result.rotation_deg:.1f}"
                cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if self.state.tracking_lost:
                cv2.putText(frame, "TRACKING LOST - PRESS R TO RESET", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            minimap = self.tile_grid.render_minimap(tile_scale=14)
            minimap = cv2.resize(minimap, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)

            frame_display = self._fit_to_height(frame, max(frame.shape[0], minimap.shape[0]))
            minimap_display = self._fit_to_height(minimap, frame_display.shape[0])

            cv2.imshow("WSI Live View", frame_display)
            cv2.imshow("WSI Tile Map", minimap_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop_event.set()
            elif key == ord("r"):
                self.state.reset_requested = True
                self.tile_grid.reset()
            elif key == ord("c"):
                self.state.calibration = None

        cv2.destroyAllWindows()

    @staticmethod
    def _fit_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
        if image.shape[0] == target_height:
            return image
        scale = target_height / float(image.shape[0])
        target_width = max(1, int(round(image.shape[1] * scale)))
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time tile-based microscope stitching demo")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile edge length in pixels")
    parser.add_argument("--cache-limit", type=int, default=500, help="Maximum number of tiles kept in RAM")
    parser.add_argument("--feather", type=int, default=24, help="Feathering distance in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = App(
        camera_index=args.camera,
        tile_size=args.tile_size,
        cache_limit=args.cache_limit,
        feather_px=args.feather,
    )
    app.start()


if __name__ == "__main__":
    main()
