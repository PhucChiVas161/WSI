from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    center_x: float
    center_y: float
    radius: float
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int

    @property
    def crop_rect(self) -> Tuple[int, int, int, int]:
        return self.crop_x, self.crop_y, self.crop_w, self.crop_h


def _ensure_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def detect_fov_circle(frame: np.ndarray) -> CalibrationResult:
    """Detect the circular microscope FOV and return a centered inscribed crop."""
    gray = _ensure_gray(frame)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(thresh)) < 127.0:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    else:
        height, width = gray.shape[:2]
        center_x = width / 2.0
        center_y = height / 2.0
        radius = min(width, height) / 2.0

    return largest_inscribed_rectangle(frame.shape, center_x, center_y, radius)


def largest_inscribed_rectangle(
    frame_shape: Tuple[int, ...],
    center_x: float,
    center_y: float,
    radius: float,
    margin_px: float = 2.0,
) -> CalibrationResult:
    """Return the largest axis-aligned square that fits inside the detected circle."""
    height, width = frame_shape[:2]
    effective_radius = max(1.0, radius - margin_px)
    side = max(1, int(round(math.sqrt(2.0) * effective_radius)))
    side = min(side, width, height)

    crop_x = int(round(center_x - side / 2.0))
    crop_y = int(round(center_y - side / 2.0))
    crop_x = max(0, min(crop_x, width - side))
    crop_y = max(0, min(crop_y, height - side))

    return CalibrationResult(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=side,
        crop_h=side,
    )


def crop_roi(frame: np.ndarray, calibration: CalibrationResult) -> np.ndarray:
    x, y, w, h = calibration.crop_rect
    return frame[y : y + h, x : x + w].copy()


def draw_calibration_overlay(frame: np.ndarray, calibration: CalibrationResult) -> np.ndarray:
    """Draw the detected circle and ROI on a preview frame."""
    overlay = frame.copy()
    cv2.circle(
        overlay,
        (int(round(calibration.center_x)), int(round(calibration.center_y))),
        int(round(calibration.radius)),
        (0, 255, 0),
        2,
    )
    x, y, w, h = calibration.crop_rect
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 180, 255), 2)
    return overlay
