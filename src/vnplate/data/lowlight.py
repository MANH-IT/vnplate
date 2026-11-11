from __future__ import annotations

import cv2
import numpy as np

LOW_LIGHT_MODES = ("auto", "clahe", "gamma", "retinex", "night", "none")


def is_low_light(frame_bgr: np.ndarray, threshold: float = 70.0) -> bool:
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


def _gamma_lut(gamma: float) -> np.ndarray:
    inv = 1.0 / max(float(gamma), 1e-6)
    table = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.float32)
    return np.clip(table, 0, 255).astype(np.uint8)


def _apply_clahe(frame_bgr: np.ndarray, clip_limit: float, tile_grid: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(float(clip_limit), 1e-3), tileGridSize=tile_grid)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _apply_gamma(frame_bgr: np.ndarray, gamma: float) -> np.ndarray:
    return cv2.LUT(frame_bgr, _gamma_lut(gamma))


def _single_scale_retinex(frame_bgr: np.ndarray, sigma: float) -> np.ndarray:
    img = frame_bgr.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img, (0, 0), max(float(sigma), 1e-3))
    retinex = np.log10(img) - np.log10(np.maximum(blur, 1.0))
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


def enhance_low_light(
    frame_bgr: np.ndarray,
    mode: str = "clahe",
    gamma: float = 1.2,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
    denoise: bool = False,
) -> np.ndarray:
    """Enhance a BGR frame for night/low-light detection.

    Modes:
    - auto: gamma theo độ sáng trung bình, sau đó CLAHE.
    - clahe: cân bằng histogram cục bộ trên kênh L (LAB).
    - gamma: tăng sáng bằng gamma LUT.
    - retinex: Single-Scale Retinex trên log domain.
    - none: trả lại frame gốc.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    mode = (mode or "clahe").lower()
    if mode == "none":
        return frame_bgr

    if mode == "auto":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        auto_gamma = 1.0 if mean >= 100 else min(3.0, 128.0 / max(mean, 1.0))
        out = _apply_gamma(frame_bgr, auto_gamma)
        out = _apply_clahe(out, clip_limit, tile_grid)
    elif mode == "gamma":
        out = _apply_gamma(frame_bgr, gamma)
    elif mode == "retinex":
        out = _single_scale_retinex(frame_bgr, sigma=max(5.0, min(frame_bgr.shape[:2]) / 20.0))
    elif mode == "night":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        auto_gamma = min(3.0, 128.0 / max(mean, 1.0))
        out = _apply_gamma(frame_bgr, auto_gamma)
        out = _apply_clahe(out, clip_limit * 1.5, tile_grid)
        out = cv2.convertScaleAbs(out, alpha=1.1, beta=5)
    else:
        out = _apply_clahe(frame_bgr, clip_limit, tile_grid)

    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
    return out
