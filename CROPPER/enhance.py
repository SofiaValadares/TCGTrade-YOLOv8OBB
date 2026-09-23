"""Deixa a carta inteira no formato fixo 63×88 mm, pronta para OCR com moldura.

Não recorta nome/HP/rodapé. O OCR usa regiões fixas em cima desta imagem.
"""

from __future__ import annotations

import cv2
import numpy as np

# Fração da imagem de saída (já com moldura). y0, y1, x0, x1.
OCR_TEMPLATE = {
    "name": (0.048, 0.128, 0.10, 0.70),
    "hp": (0.048, 0.128, 0.70, 0.96),
    "footer": (0.90, 0.985, 0.04, 0.55),
}

_FRAME_PAD = 0.02
_FRAME_COLOR = (248, 248, 248)


def inset_quad(quad: np.ndarray, frac: float) -> np.ndarray:
    """Encolhe o quadrilátero em direção ao centro (tira um pouco de bolso)."""
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if frac <= 0:
        return pts
    center = pts.mean(axis=0)
    return (center + (pts - center) * (1.0 - frac)).astype(np.float32)


def suppress_glare(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    glare = (hsv[:, :, 2] > 242) & (hsv[:, :, 1] < 70)
    if int(glare.sum()) < 40:
        return bgr
    med = cv2.medianBlur(bgr, 5)
    out = bgr.copy()
    out[glare] = med[glare]
    return out


def flatten_light(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    luma, a, b = cv2.split(lab)
    shade = cv2.GaussianBlur(luma, (0, 0), 28)
    shade = np.maximum(shade, 1)
    flat = cv2.normalize(
        (luma.astype(np.float32) / shade.astype(np.float32)) * 128.0,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([flat, a, b]), cv2.COLOR_LAB2BGR)


def enhance_card(bgr: np.ndarray) -> np.ndarray:
    """Luz mais uniforme e texto um pouco mais nítido — carta inteira."""
    out = flatten_light(suppress_glare(bgr))
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    luma, a, b = cv2.split(lab)
    luma = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(luma)
    out = cv2.cvtColor(cv2.merge([luma, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 0.8)
    return cv2.addWeighted(out, 1.22, blur, -0.22, 0)


def apply_frame(bgr: np.ndarray, pad_frac: float = _FRAME_PAD) -> np.ndarray:
    """Encaixa a carta numa moldura clara, tamanho final inalterado (63×88 mm)."""
    h, w = bgr.shape[:2]
    pad_x = max(2, int(round(w * pad_frac)))
    pad_y = max(2, int(round(h * pad_frac)))
    inner_w = w - 2 * pad_x
    inner_h = h - 2 * pad_y
    if inner_w < 8 or inner_h < 8:
        return bgr
    inner = cv2.resize(bgr, (inner_w, inner_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), _FRAME_COLOR, dtype=np.uint8)
    canvas[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w] = inner
    return canvas


def draw_ocr_template(bgr: np.ndarray) -> np.ndarray:
    """Desenha a moldura de OCR (só visualização)."""
    vis = bgr.copy()
    h, w = vis.shape[:2]
    colors = {"name": (40, 200, 40), "hp": (30, 140, 255), "footer": (220, 160, 40)}
    for key, (y0, y1, x0, x1) in OCR_TEMPLATE.items():
        p1 = (int(x0 * w), int(y0 * h))
        p2 = (int(x1 * w), int(y1 * h))
        cv2.rectangle(vis, p1, p2, colors[key], 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            key,
            (p1[0] + 4, p1[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[key],
            1,
            cv2.LINE_AA,
        )
    return vis


def to_bw(bgr: np.ndarray) -> np.ndarray:
    """Carta inteira em preto e branco (contraste alto, para OCR)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    return gray


def prepare_card(bgr: np.ndarray, enhance: bool = True, frame: bool = False) -> np.ndarray:
    out = enhance_card(bgr) if enhance else bgr
    return apply_frame(out) if frame else out
