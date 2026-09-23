"""Retifica uma carta a partir de um quadrilátero OBB (perspectiva estilo scanner).

A saída respeita a proporção física de uma carta de Pokémon: 63 mm × 88 mm.

O OBB do YOLO é um retângulo girado e costuma incluir um pouco de fundo/plástico.
O passo `refine` (desligado por padrão) tentava colar o OBB na borda por
gradiente — para OCR isso corta nome ou deixa vizinha. O recorte usa o OBB
+ inset fixo + CLAHE/nitidez (`enhance.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from enhance import enhance_ocr, inset_quad, ocr_bands

CARD_WIDTH_MM = 63.0
CARD_HEIGHT_MM = 88.0
MM_PER_INCH = 25.4
DEFAULT_DPI = 300

# Folga no primeiro warp para o snap ter margem de procurar a borda.
_PAD_SCALE = 1.18
_SEARCH_FRAC = 0.28
_MIN_AREA_FRAC = 0.55
_MAX_AREA_FRAC = 0.97


def card_size_px(dpi: int = DEFAULT_DPI) -> tuple[int, int]:
    """Largura × altura em pixels para 63 mm × 88 mm no DPI pedido."""
    width = max(1, round(CARD_WIDTH_MM / MM_PER_INCH * dpi))
    height = max(1, round(CARD_HEIGHT_MM / MM_PER_INCH * dpi))
    return width, height


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos como TL, TR, BR, BL (sentido horário, y para baixo)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)

    edge = pts[1] - pts[0]
    to_last = pts[-1] - pts[0]
    cross = float(edge[0] * to_last[1] - edge[1] * to_last[0])
    if cross < 0:
        pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float32)
    return pts.astype(np.float32)


def orient_portrait(quad: np.ndarray) -> np.ndarray:
    """Gira o mapeamento 90° se a caixa estiver deitada, para a saída ser 63×88 (retrato)."""
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if width > height:
        quad = np.array([quad[1], quad[2], quad[3], quad[0]], dtype=np.float32)
    return quad


def expand_quad(quad: np.ndarray, scale: float = _PAD_SCALE) -> np.ndarray:
    center = np.asarray(quad, dtype=np.float32).reshape(4, 2).mean(axis=0)
    return (center + (quad - center) * scale).astype(np.float32)


def _destination(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )


def warp_quad(
    image: np.ndarray,
    quad: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    src = orient_portrait(order_corners(quad))
    matrix = cv2.getPerspectiveTransform(src, _destination(width, height))
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _line_from_points(xs: np.ndarray, ys: np.ndarray, x_from_y: bool) -> tuple[float, float] | None:
    if len(xs) < 20:
        return None
    if x_from_y:
        med = float(np.median(xs))
        keep = np.abs(xs - med) <= max(8.0, 0.04 * (xs.max() - xs.min() + 1))
        ys, xs = ys[keep], xs[keep]
        if len(xs) < 20:
            return None
        a, b = np.polyfit(ys.astype(np.float64), xs.astype(np.float64), 1)
    else:
        med = float(np.median(ys))
        keep = np.abs(ys - med) <= max(8.0, 0.04 * (ys.max() - ys.min() + 1))
        xs, ys = xs[keep], ys[keep]
        if len(ys) < 20:
            return None
        a, b = np.polyfit(xs.astype(np.float64), ys.astype(np.float64), 1)
    return float(a), float(b)


def _intersect(a1: float, b1: float, a2: float, b2: float, x_from_y_first: bool) -> np.ndarray:
    """Interseção de x=a1*y+b1 com y=a2*x+b2 (ou o inverso)."""
    if x_from_y_first:
        # x = a1 y + b1 ; y = a2 x + b2
        y = (a2 * b1 + b2) / (1.0 - a2 * a1)
        x = a1 * y + b1
    else:
        x = (a2 * b1 + b2) / (1.0 - a2 * a1)
        y = a1 * x + b1
    return np.array([x, y], dtype=np.float32)


def _smooth1d(values: np.ndarray, k: int = 21) -> np.ndarray:
    k = k if k % 2 == 1 else k + 1
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def _step_index(profile: np.ndarray, band: int, from_start: bool, falling: bool) -> int | None:
    """Índice do degrau tecido→carta no terço externo — o mais externo, não o mais forte."""
    if profile.size < band + 5:
        return None
    deriv = np.diff(_smooth1d(profile))
    skip = max(8, band // 12)
    if from_start:
        region = deriv[skip:band]
        origin = skip
    else:
        end = len(deriv) - skip
        start = max(0, len(deriv) - band)
        region = deriv[start:end]
        origin = start
    if region.size < 5:
        return None
    peak = float(np.max(np.abs(region)))
    thresh = max(0.45, 0.22 * peak)
    if falling:
        hits = np.flatnonzero(region <= -thresh)
    else:
        hits = np.flatnonzero(region >= thresh)
    if hits.size == 0:
        return None
    rel = int(hits[0] if from_start else hits[-1])
    return origin + rel


def refine_quad(
    image: np.ndarray,
    quad: np.ndarray,
    dpi: int = DEFAULT_DPI,
) -> np.ndarray | None:
    """Pós-anotação: cola os 4 cantos na borda impressa, na imagem original.

    A caixa OBB (anotação/detector) é um retângulo e costuma incluir o bolso.
    Aqui a imagem é warpeada com folga, o snap acha as 4 bordas, e os cantos
    voltam para o sistema de coordenadas da foto. `None` se o snap falhar.
    """
    h, w = image.shape[:2]
    src = orient_portrait(order_corners(quad))
    width, height = card_size_px(dpi)
    pad_w = int(round(width * _PAD_SCALE))
    pad_h = int(round(height * _PAD_SCALE))
    loose_src = expand_quad(src, _PAD_SCALE)
    dst = _destination(pad_w, pad_h)
    matrix = cv2.getPerspectiveTransform(loose_src, dst)
    loose = cv2.warpPerspective(
        image,
        matrix,
        (pad_w, pad_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    snapped = snap_card_quad(loose)
    if snapped is None:
        return None
    pts = cv2.perspectiveTransform(snapped.reshape(1, 4, 2), np.linalg.inv(matrix))
    pts = pts.reshape(4, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    if float(cv2.contourArea(pts.reshape(-1, 1, 2))) < 16:
        return None
    return order_corners(pts.astype(np.float32))


def snap_card_quad(warped: np.ndarray) -> np.ndarray | None:
    """Acha a carta via perfil de brilho (tecido claro → borda mais escura)."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    band_x = max(24, int(w * _SEARCH_FRAC))
    band_y = max(24, int(h * _SEARCH_FRAC))
    y0, y1 = int(h * 0.18), int(h * 0.82)
    x0, x1 = int(w * 0.18), int(w * 0.82)

    col = gray[y0:y1, :].mean(axis=0)
    row = gray[:, x0:x1].mean(axis=1)
    left_x = _step_index(col, band_x, from_start=True, falling=True)
    right_x = _step_index(col, band_x, from_start=False, falling=False)
    top_y = _step_index(row, band_y, from_start=True, falling=True)
    bot_y = _step_index(row, band_y, from_start=False, falling=False)
    if None in (left_x, right_x, top_y, bot_y):
        return None

    gx = cv2.Sobel(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F, 0, 1, ksize=3)
    win = 14

    def line_vertical(x_est: int, want_negative: bool) -> tuple[float, float] | None:
        x0b, x1b = max(0, x_est - win), min(w, x_est + win + 1)
        mag = -gx[:, x0b:x1b] if want_negative else gx[:, x0b:x1b]
        xs, ys = [], []
        for y in range(y0, y1):
            vec = mag[y]
            idx = int(np.argmax(vec))
            if vec[idx] < 8:
                continue
            xs.append(x0b + idx)
            ys.append(y)
        return _line_from_points(np.array(xs, np.float32), np.array(ys, np.float32), True)

    def line_horizontal(y_est: int, want_negative: bool) -> tuple[float, float] | None:
        y0b, y1b = max(0, y_est - win), min(h, y_est + win + 1)
        mag = -gy[y0b:y1b, :] if want_negative else gy[y0b:y1b, :]
        xs, ys = [], []
        for x in range(x0, x1):
            vec = mag[:, x]
            idx = int(np.argmax(vec))
            if vec[idx] < 8:
                continue
            xs.append(x)
            ys.append(y0b + idx)
        return _line_from_points(np.array(xs, np.float32), np.array(ys, np.float32), False)

    left = line_vertical(left_x, True) or (0.0, float(left_x))
    right = line_vertical(right_x, False) or (0.0, float(right_x))
    top = line_horizontal(top_y, True) or (0.0, float(top_y))
    bottom = line_horizontal(bot_y, False) or (0.0, float(bot_y))

    a_l, b_l = left
    a_r, b_r = right
    a_t, b_t = top
    a_b, b_b = bottom
    try:
        tl = _intersect(a_l, b_l, a_t, b_t, True)
        tr = _intersect(a_r, b_r, a_t, b_t, True)
        br = _intersect(a_r, b_r, a_b, b_b, True)
        bl = _intersect(a_l, b_l, a_b, b_b, True)
    except ZeroDivisionError:
        return None

    quad = np.stack([tl, tr, br, bl]).astype(np.float32)
    area = float(cv2.contourArea(quad.reshape(-1, 1, 2)))
    img_area = float(w * h)
    if not (_MIN_AREA_FRAC * img_area <= area <= _MAX_AREA_FRAC * img_area):
        return None
    if tl[0] >= tr[0] - 8 or tl[1] >= bl[1] - 8:
        return None
    return order_corners(quad)


def rectify_card(
    image: np.ndarray,
    quad: np.ndarray,
    dpi: int = DEFAULT_DPI,
    refine: bool = False,
    inset: float = 0.02,
    enhance: bool = True,
) -> np.ndarray:
    """Perspectiva a partir do OBB. `inset` encolhe a caixa; `enhance` é para OCR."""
    width, height = card_size_px(dpi)
    src_quad = orient_portrait(order_corners(quad))
    if refine:
        snapped = refine_quad(image, src_quad, dpi=dpi)
        if snapped is not None:
            src_quad = orient_portrait(order_corners(snapped))
    if inset > 0:
        src_quad = inset_quad(src_quad, inset)
    warped = warp_quad(image, src_quad, width, height)
    return enhance_ocr(warped) if enhance else warped


@dataclass
class CroppedCard:
    image: np.ndarray
    conf: float
    quad: np.ndarray
    index: int
    bands: dict = field(default_factory=dict)


def quads_from_obb_result(result) -> list[tuple[np.ndarray, float]]:
    """Extrai (quadrilátero 4×2, confidence) de um `ultralytics.engine.results.Results`."""
    if result.obb is None or len(result.obb) == 0:
        return []
    xy = result.obb.xyxyxyxy.cpu().numpy()
    confs = result.obb.conf.cpu().numpy()
    return [(quad.reshape(4, 2), float(conf)) for quad, conf in zip(xy, confs)]


def crop_result(
    result,
    dpi: int = DEFAULT_DPI,
    image: np.ndarray | None = None,
    refine: bool = False,
    inset: float = 0.02,
    enhance: bool = True,
) -> list[CroppedCard]:
    """Gera uma imagem retificada por detecção OBB no `result`."""
    bgr = image if image is not None else result.orig_img
    if bgr is None:
        raise ValueError("Sem imagem de origem: passe `image=` ou um result com orig_img.")
    cropped: list[CroppedCard] = []
    for i, (quad, conf) in enumerate(quads_from_obb_result(result)):
        used = order_corners(quad)
        image_out = rectify_card(
            bgr, used, dpi=dpi, refine=refine, inset=inset, enhance=enhance
        )
        cropped.append(
            CroppedCard(
                image=image_out,
                conf=conf,
                quad=used,
                index=i,
                bands=ocr_bands(image_out),
            )
        )
    return cropped
