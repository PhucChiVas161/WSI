from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import cv2
import numpy as np


@dataclass
class StitchResult:
    success: bool
    homography: Optional[np.ndarray]
    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    matches: int = 0
    inliers: int = 0
    confidence: float = 0.0
    reason: str = ""


class ORBStitchingEngine:
    def __init__(
        self,
        nfeatures: int = 2500,
        matcher_type: str = "bf",
        ratio_test: float = 0.75,
        min_matches: int = 12,
        ransac_threshold: float = 4.0,
    ) -> None:
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher_type = matcher_type.lower()
        self.ratio_test = ratio_test
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.matcher = self._create_matcher()

    def _create_matcher(self):
        if self.matcher_type == "flann":
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=64)
            return cv2.FlannBasedMatcher(index_params, search_params)
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @staticmethod
    def _ensure_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def estimate_motion(self, previous_frame: np.ndarray, current_frame: np.ndarray) -> StitchResult:
        prev_gray = self._ensure_gray(previous_frame)
        curr_gray = self._ensure_gray(current_frame)

        kp_prev, des_prev = self.orb.detectAndCompute(prev_gray, None)
        kp_curr, des_curr = self.orb.detectAndCompute(curr_gray, None)
        if des_prev is None or des_curr is None:
            return StitchResult(False, None, reason="not_enough_features")

        if self.matcher_type == "flann":
            des_prev = np.asarray(des_prev, dtype=np.uint8)
            des_curr = np.asarray(des_curr, dtype=np.uint8)

        raw_matches = self.matcher.knnMatch(des_prev, des_curr, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.min_matches:
            return StitchResult(False, None, matches=len(good_matches), reason="not_enough_matches")

        prev_pts = np.float32([kp_prev[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, self.ransac_threshold)
        if homography is None or mask is None:
            return StitchResult(False, None, matches=len(good_matches), reason="homography_failed")

        inliers = int(mask.ravel().sum())
        confidence = inliers / max(1, len(good_matches))
        dx = float(homography[0, 2])
        dy = float(homography[1, 2])
        rotation_deg = math.degrees(math.atan2(float(homography[1, 0]), float(homography[0, 0])))

        return StitchResult(
            True,
            homography=homography,
            dx=dx,
            dy=dy,
            rotation_deg=rotation_deg,
            matches=len(good_matches),
            inliers=inliers,
            confidence=confidence,
        )

    @staticmethod
    def invert_transform(transform: np.ndarray) -> np.ndarray:
        return np.linalg.inv(transform)

    @staticmethod
    def compose(world_from_previous: np.ndarray, previous_to_current: np.ndarray) -> np.ndarray:
        return world_from_previous @ np.linalg.inv(previous_to_current)

    @staticmethod
    def warp_into_frame(frame: np.ndarray, homography: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        width, height = size
        return cv2.warpPerspective(frame, homography, (width, height), flags=cv2.INTER_LINEAR)
