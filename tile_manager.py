from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import time
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


class TileStatus(str, Enum):
    EMPTY = "EMPTY"
    SCANNING = "SCANNING"
    LOCKED = "LOCKED"


@dataclass
class Tile:
    image: Optional[np.ndarray] = None
    coverage: Optional[np.ndarray] = None
    status: TileStatus = TileStatus.EMPTY
    hit_count: int = 0
    quality_score: float = 0.0
    persisted_path: Optional[Path] = None
    last_access: float = field(default_factory=time.time)


class SparseTileGrid:
    def __init__(
        self,
        tile_size: int = 512,
        cache_limit: int = 500,
        lock_coverage_threshold: float = 0.92,
        lock_hit_threshold: int = 3,
        feather_px: int = 24,
        spill_directory: Optional[Path] = None,
    ) -> None:
        self.tile_size = tile_size
        self.cache_limit = cache_limit
        self.lock_coverage_threshold = lock_coverage_threshold
        self.lock_hit_threshold = lock_hit_threshold
        self.feather_px = feather_px
        self.tiles: Dict[Tuple[int, int], Tile] = {}
        self.lru: OrderedDict[Tuple[int, int], None] = OrderedDict()
        self.spill_directory = spill_directory or Path(tempfile.gettempdir()) / "wsi_tile_cache"
        self.spill_directory.mkdir(parents=True, exist_ok=True)
        self._min_row = 0
        self._max_row = -1
        self._min_col = 0
        self._max_col = -1

    def _touch(self, key: Tuple[int, int]) -> None:
        self.lru.pop(key, None)
        self.lru[key] = None
        tile = self.tiles[key]
        tile.last_access = time.time()

    def _enforce_cache_limit(self) -> None:
        while len(self.tiles) > self.cache_limit and self.lru:
            key, _ = self.lru.popitem(last=False)
            tile = self.tiles.get(key)
            if tile is None:
                continue
            self._spill_tile(key, tile)

    def _spill_tile(self, key: Tuple[int, int], tile: Tile) -> None:
        if tile.image is None and tile.coverage is None:
            return
        path = self.spill_directory / f"tile_{key[0]}_{key[1]}.npz"
        np.savez_compressed(
            path,
            image=tile.image,
            coverage=tile.coverage,
            status=tile.status.value,
            hit_count=tile.hit_count,
            quality_score=tile.quality_score,
        )
        tile.persisted_path = path
        tile.image = None
        tile.coverage = None

    def _load_tile(self, key: Tuple[int, int], tile: Tile) -> None:
        if tile.persisted_path is None or not tile.persisted_path.exists():
            return
        with np.load(tile.persisted_path, allow_pickle=True) as data:
            image = data["image"]
            coverage = data["coverage"]
            status_value = str(data["status"])
            tile.image = None if image is None else image
            tile.coverage = None if coverage is None else coverage
            tile.status = TileStatus(status_value)
            tile.hit_count = int(data["hit_count"])
            tile.quality_score = float(data["quality_score"])

    def get_tile(self, row: int, col: int) -> Tile:
        key = (row, col)
        tile = self.tiles.get(key)
        if tile is None:
            tile = Tile(status=TileStatus.SCANNING)
            self.tiles[key] = tile
        elif tile.image is None and tile.coverage is None and tile.persisted_path is not None:
            self._load_tile(key, tile)
        self._touch(key)
        self._update_bounds(row, col)
        self._enforce_cache_limit()
        return tile

    def _update_bounds(self, row: int, col: int) -> None:
        if self._max_row < self._min_row:
            self._min_row = self._max_row = row
            self._min_col = self._max_col = col
            return
        self._min_row = min(self._min_row, row)
        self._max_row = max(self._max_row, row)
        self._min_col = min(self._min_col, col)
        self._max_col = max(self._max_col, col)

    @staticmethod
    def _feather_mask(height: int, width: int, feather_px: int) -> np.ndarray:
        if feather_px <= 0:
            return np.ones((height, width), dtype=np.float32)
        mask = np.ones((height, width), dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, feather_px, dtype=np.float32)
        mask[:feather_px, :] *= ramp[:, None]
        mask[-feather_px:, :] *= ramp[::-1][:, None]
        mask[:, :feather_px] *= ramp[None, :]
        mask[:, -feather_px:] *= ramp[::-1][None, :]
        return np.clip(mask, 0.0, 1.0)

    @staticmethod
    def _tile_warp_matrix(world_from_frame: np.ndarray, tile_x: int, tile_y: int) -> np.ndarray:
        translation = np.array(
            [[1.0, 0.0, -float(tile_x)], [0.0, 1.0, -float(tile_y)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return translation @ world_from_frame

    def insert_frame(self, frame: np.ndarray, world_from_frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        corners = np.array(
            [[[0.0, 0.0]], [[float(width), 0.0]], [[float(width), float(height)]], [[0.0, float(height)]]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(corners, world_from_frame)
        min_x = int(np.floor(mapped[:, 0, 0].min()))
        max_x = int(np.ceil(mapped[:, 0, 0].max()))
        min_y = int(np.floor(mapped[:, 0, 1].min()))
        max_y = int(np.ceil(mapped[:, 0, 1].max()))

        first_col = min_x // self.tile_size
        last_col = max_x // self.tile_size
        first_row = min_y // self.tile_size
        last_row = max_y // self.tile_size

        feather_mask = self._feather_mask(height, width, self.feather_px)
        for row in range(first_row, last_row + 1):
            for col in range(first_col, last_col + 1):
                tile = self.get_tile(row, col)
                if tile.status == TileStatus.LOCKED:
                    continue

                tile_x = col * self.tile_size
                tile_y = row * self.tile_size
                local_h = self._tile_warp_matrix(world_from_frame, tile_x, tile_y)
                warped = cv2.warpPerspective(frame, local_h, (self.tile_size, self.tile_size), flags=cv2.INTER_LINEAR)
                warped_mask = cv2.warpPerspective(
                    feather_mask,
                    local_h,
                    (self.tile_size, self.tile_size),
                    flags=cv2.INTER_LINEAR,
                )

                self._blend_into_tile(tile, warped, warped_mask)
                tile.hit_count += 1
                tile.coverage = None if tile.coverage is None else tile.coverage
                tile.quality_score = float(np.mean(tile.coverage)) if tile.coverage is not None else 0.0
                if tile.quality_score >= self.lock_coverage_threshold and tile.hit_count >= self.lock_hit_threshold:
                    tile.status = TileStatus.LOCKED
                else:
                    tile.status = TileStatus.SCANNING

    @staticmethod
    def _blend_into_tile(tile: Tile, patch: np.ndarray, patch_mask: np.ndarray) -> None:
        if tile.image is None:
            tile.image = np.zeros_like(patch)
        if tile.coverage is None:
            tile.coverage = np.zeros(patch_mask.shape, dtype=np.float32)

        existing = tile.image.astype(np.float32)
        existing_weight = tile.coverage[..., None]
        new_weight = np.clip(patch_mask[..., None], 0.0, 1.0)
        total_weight = existing_weight + new_weight

        blended = np.where(
            total_weight > 0.0,
            (existing * existing_weight + patch.astype(np.float32) * new_weight) / np.maximum(total_weight, 1e-6),
            existing,
        )
        tile.image = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        tile.coverage = np.clip(np.maximum(tile.coverage, patch_mask), 0.0, 1.0)

    def render_minimap(self, tile_scale: int = 12) -> np.ndarray:
        if not self.tiles:
            return np.zeros((tile_scale * 4, tile_scale * 4, 3), dtype=np.uint8)

        rows = self._max_row - self._min_row + 1
        cols = self._max_col - self._min_col + 1
        canvas = np.zeros((rows * tile_scale, cols * tile_scale, 3), dtype=np.uint8)

        for (row, col), tile in self.tiles.items():
            y0 = (row - self._min_row) * tile_scale
            x0 = (col - self._min_col) * tile_scale
            if tile.status == TileStatus.LOCKED:
                color = (40, 180, 60)
            elif tile.status == TileStatus.SCANNING:
                color = (200, 140, 30)
            else:
                color = (50, 50, 50)
            cv2.rectangle(canvas, (x0, y0), (x0 + tile_scale - 1, y0 + tile_scale - 1), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + tile_scale - 1, y0 + tile_scale - 1), (25, 25, 25), 1)

        return canvas

    def reset(self) -> None:
        self.tiles.clear()
        self.lru.clear()
        self._min_row = 0
        self._max_row = -1
        self._min_col = 0
        self._max_col = -1
