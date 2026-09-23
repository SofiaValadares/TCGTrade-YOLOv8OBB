"""Tratamentos na carta já retificada, pensados para OCR (nome, HP, número).

Não procura borda na foto original e não mexe no OBB.
"""

from __future__ import annotations

import cv2
import numpy as np

# Fração do layout TCG (carta retrato já warpeada).
_NAME_Y = (0.035, 0.125)
_NAME_X = (0.06, 0.78)
_FOOTER_Y = (0.915, 0.985)
_FOOTER_X = (0.02, 0.62)


def inset_quad(quad: np.ndarray, frac: float) -> np.ndarray:
    """Encolhe o quadrilátero em direção ao centro (tira bolso/vizinha sem procurar borda)."""
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if frac <= 0:
        return pts
    center = pts.mean(axis=0)
    return (center + (pts - center) * (1.0 - frac)).astype(np.float32)


def enhance_ocr(bgr: np.ndarray) -> np.ndarray:
    """Equaliza luz (CLAHE) e aumenta nitidez do texto. Sem recortar."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    luma, a, b = cv2.split(lab)
    luma = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luma)
    out = cv2.cvtColor(cv2.merge([luma, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.1)
    return cv2.addWeighted(out, 1.45, blur, -0.45, 0)


def ocr_bands(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Faixas fixas do layout Pokémon: nome (topo) e número da set (rodapé)."""
    h, w = bgr.shape[:2]

    def crop(y0: float, y1: float, x0: float, x1: float) -> np.ndarray:
        return bgr[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]

    return {
        "name": crop(*_NAME_Y, *_NAME_X),
        "footer": crop(*_FOOTER_Y, *_FOOTER_X),
    }
