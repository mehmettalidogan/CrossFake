"""Self-Blended Images (Shiohara & Yamasaki, CVPR 2022) training synthesizer.

Creates pseudo-fakes from a single real face by blending two differently
augmented copies with a deformed convex-hull / ellipse mask. Landmark JSON
sidecar files written by extract_faces.py are used when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_landmarks(image_path: str | Path) -> np.ndarray | None:
    side = Path(image_path).with_suffix(".json")
    if not side.exists():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        pts = data.get("landmarks") or data.get("kps")
        if pts is None:
            return None
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        return arr
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _convex_hull_mask(shape_hw: tuple[int, int], landmarks: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)
    hull = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)
    return mask


def _ellipse_mask(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, int(h * 0.48))
    axes = (int(w * 0.38), int(h * 0.48))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    return mask


def _deform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = mask.shape
    # Elastic-like grid warp of the blend boundary.
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    amp = rng.uniform(4.0, 18.0)
    freq = rng.uniform(1.5, 4.0)
    dx = amp * np.sin(2 * np.pi * grid_y / max(h, 1) * freq + rng.uniform(0, 6))
    dy = amp * np.cos(2 * np.pi * grid_x / max(w, 1) * freq + rng.uniform(0, 6))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    warped = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    k = int(rng.integers(7, 31)) | 1
    warped = cv2.GaussianBlur(warped, (k, k), 0)
    return np.clip(warped, 0.0, 1.0)


def _color_transfer(src: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = src.astype(np.float32)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.75, 1.25), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    return out


def _frequency_mix(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Downsample / blur then resize to mimic compression & frequency mismatch."""
    h, w = img.shape[:2]
    scale = rng.uniform(0.25, 0.7)
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5, 7]))
        back = cv2.GaussianBlur(back, (k, k), 0)
    return back


def _affine_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    max_shift = 0.04
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = src + rng.uniform(-max_shift, max_shift, size=src.shape).astype(np.float32) * np.array(
        [w, h], dtype=np.float32
    )
    m = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def synthesize_sbi(
    image_rgb: np.ndarray,
    landmarks: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a self-blended fake (uint8 RGB) from a real face crop."""
    rng = rng or np.random.default_rng()
    img = image_rgb.astype(np.float32)
    h, w = img.shape[:2]

    source = _color_transfer(img, rng)
    source = _frequency_mix(source, rng)
    target = _affine_jitter(img, rng)
    if rng.random() < 0.5:
        target = _color_transfer(target, rng)

    if landmarks is not None and len(landmarks) >= 3:
        # Scale landmarks if they were stored in original crop size.
        mask = _convex_hull_mask((h, w), landmarks)
    else:
        mask = _ellipse_mask((h, w))
    mask = _deform_mask(mask, rng)[..., None]

    blended = source * mask + target * (1.0 - mask)
    return np.clip(blended, 0, 255).astype(np.uint8)
