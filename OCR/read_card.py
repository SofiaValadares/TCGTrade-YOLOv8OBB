"""Detecta regiões (nome / número / coleção) na carta recortada e lê o texto."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

# Roboflow: 0=colection (typo), 1=name, 2=number
CLASS_TO_FIELD = {0: "collection", 1: "name", 2: "number"}
FIELD_TO_CLASS = {v: k for k, v in CLASS_TO_FIELD.items()}

FALLBACK_TEMPLATE = {
    "name": (0.048, 0.128, 0.10, 0.68),
    "number": (0.915, 0.995, 0.14, 0.48),
    "collection": (0.915, 0.995, 0.48, 0.92),
}

_NAME_ALLOW = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "ÁÉÍÓÚáéíóúÃÕãõÂÊÔâêôÀàÇçÜü '-"
)
# Letras que o EasyOCR troca por dígitos entram na leitura e são corrigidas em clean_number.
_NUMBER_ALLOW = "0123456789/OoQqDdIiLl|ZzSsGgBb"
_NUMBER_TRANSLATE = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "q": "0",
        "D": "0",
        "I": "1",
        "i": "1",
        "L": "1",
        "l": "1",
        "|": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "G": "6",
        "b": "6",
        "B": "8",
        "g": "9",
    }
)
_COLLECTION_ALLOW = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "ÁÉÍÓÚáéíóúÃÕãõÂÊÔâêô '-"
)
_FIELD_ALLOW = {
    "name": _NAME_ALLOW,
    "number": _NUMBER_ALLOW,
    "collection": _COLLECTION_ALLOW,
}
_WARP_PAD = {"name": 0.18, "number": 0.12, "collection": 0.14}


@dataclass
class Region:
    field: str
    image: np.ndarray
    conf: float
    quad: np.ndarray
    text: str = ""
    source: str = "det"
    ocr_conf: float = 0.0


@dataclass
class CardRead:
    name: str = ""
    number: str = ""
    collection: str = ""
    regions: dict[str, Region] = field(default_factory=dict)


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)
    edge = pts[1] - pts[0]
    to_last = pts[-1] - pts[0]
    if float(edge[0] * to_last[1] - edge[1] * to_last[0]) < 0:
        pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float32)
    return pts.astype(np.float32)


def warp_quad(image: np.ndarray, quad: np.ndarray, pad: float = 0.12) -> np.ndarray:
    src = order_corners(quad)
    center = src.mean(axis=0)
    src = center + (src - center) * (1.0 + pad)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    w, h = max(w, 8), max(h, 8)
    if h > w * 1.4:
        src = np.array([src[1], src[2], src[3], src[0]], dtype=np.float32)
        w, h = h, w
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_CUBIC)


def crop_template(image: np.ndarray, field: str) -> np.ndarray:
    y0, y1, x0, x1 = FALLBACK_TEMPLATE[field]
    h, w = image.shape[:2]
    crop = image[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
    return crop if crop.size else image


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_strip(image: np.ndarray, min_h: int = 112) -> np.ndarray:
    """Amplia e aumenta contraste — faixas de binder são baixas e JPEG-comprimidas."""
    gray = _gray(image)
    h, w = gray.shape[:2]
    if h < min_h:
        scale = min_h / max(h, 1)
        gray = cv2.resize(
            gray,
            (max(8, int(round(w * scale))), min_h),
            interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(4, 4)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 0.9)
    return cv2.addWeighted(gray, 1.55, blur, -0.55, 0)


def _strip_variants(gray: np.ndarray) -> list[np.ndarray]:
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [gray, 255 - gray, otsu, 255 - otsu]


def _readtext(reader, rgb: np.ndarray, allowlist: str):
    attempts = (
        {"detail": 1, "paragraph": False, "allowlist": allowlist, "mag_ratio": 1.8},
        {"detail": 1, "paragraph": False, "allowlist": allowlist},
        {"detail": 1, "paragraph": False},
    )
    last_items = []
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            items = reader.readtext(rgb, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        text, _conf = _parse_ocr_items(items)
        if text.strip():
            return items
        last_items = items
    if last_error and not last_items:
        return reader.readtext(rgb)
    return last_items


def _parse_ocr_items(items) -> tuple[str, float]:
    parts: list[str] = []
    scores: list[float] = []
    for item in items or []:
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
            continue
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = str(item[1]).strip()
        conf = float(item[2]) if len(item) > 2 else 0.0
        if text:
            parts.append(text)
            scores.append(conf)
    joined = " ".join(parts)
    mean = float(np.mean(scores)) if scores else 0.0
    return joined, mean


def _format_number(left: str, right: str) -> str:
    return f"{int(left):03d}/{int(right)}"


def _split_number_digits(digits: str) -> tuple[str, str] | None:
    """Remonta NNN/NNN quando a barra some ou é lida como 1."""
    if len(digits) == 7 and digits[3] == "1":
        return digits[:3], digits[4:]
    if len(digits) == 6:
        return digits[:3], digits[3:]
    if len(digits) == 5:
        return digits[:2], digits[2:]
    return None


def clean_number(text: str) -> str:
    raw = (text or "").translate(_NUMBER_TRANSLATE)
    raw = raw.replace(" ", "").replace("\\", "/").replace("⁄", "/")
    match = re.search(r"(\d{1,3})/(\d{2,3})", raw)
    if match:
        return _format_number(match.group(1), match.group(2))
    parts = _split_number_digits(re.sub(r"\D", "", raw))
    if parts:
        return _format_number(*parts)
    return ""


def clean_name(text: str) -> str:
    text = re.sub(r"^[^A-Za-zÁ-ú]+", "", text)
    text = re.sub(r"[^A-Za-zÁ-ú0-9 '\-]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -'")
    return text


def clean_collection(text: str) -> str:
    """Fica o código da coleção: 3 a 7 letras ou dígitos, como MEW, OBF ou 151."""
    code = re.sub(r"[^A-Za-z0-9]+", "", text or "")
    if 3 <= len(code) <= 7:
        return code.upper()
    return ""


def _clean(field: str, text: str) -> str:
    if field == "number":
        return clean_number(text)
    if field == "collection":
        return clean_collection(text)
    return clean_name(text)


def ocr_image(reader, image: np.ndarray, field: str = "name") -> tuple[str, float, np.ndarray]:
    min_h = 80 if field == "collection" else 112
    enhanced = enhance_strip(image, min_h=min_h)
    allow = _FIELD_ALLOW[field]

    def _run(view: np.ndarray) -> tuple[str, float]:
        rgb = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
        text, conf = _parse_ocr_items(_readtext(reader, rgb, allow))
        return _clean(field, text), conf

    text, conf = _run(enhanced)
    if field == "collection" and conf < 0.35:
        text = ""
    if text:
        return text, conf, enhanced
    if field == "name":
        inv = 255 - enhanced
        text, conf = _run(inv)
        if text:
            return text, conf, inv
    if field == "number":
        best_text, best_conf, best_view = "", -1.0, enhanced
        for view in _strip_variants(enhanced)[1:]:
            cand, cconf = _run(view)
            if cand and cconf > best_conf:
                best_text, best_conf, best_view = cand, cconf, view
        return best_text, max(best_conf, 0.0), best_view
    return "", max(conf, 0.0), enhanced


def regions_from_obb(result, image: np.ndarray | None = None, conf_min: float = 0.25) -> dict[str, Region]:
    bgr = image if image is not None else result.orig_img
    found: dict[str, Region] = {}
    if result.obb is None or len(result.obb) == 0:
        return found
    xy = result.obb.xyxyxyxy.cpu().numpy()
    cls = result.obb.cls.cpu().numpy().astype(int)
    confs = result.obb.conf.cpu().numpy()
    for quad, c, cf in zip(xy, cls, confs):
        if cf < conf_min:
            continue
        field_name = CLASS_TO_FIELD.get(int(c))
        if field_name is None:
            continue
        if field_name in found and found[field_name].conf >= float(cf):
            continue
        found[field_name] = Region(
            field=field_name,
            image=warp_quad(bgr, quad.reshape(4, 2), pad=_WARP_PAD[field_name]),
            conf=float(cf),
            quad=quad.reshape(4, 2),
            source="det",
        )
    return found


def fill_missing(image: np.ndarray, regions: dict[str, Region]) -> dict[str, Region]:
    for field_name in ("name", "number", "collection"):
        if field_name in regions:
            continue
        crop = crop_template(image, field_name)
        h, w = image.shape[:2]
        y0, y1, x0, x1 = FALLBACK_TEMPLATE[field_name]
        quad = np.array(
            [[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]],
            dtype=np.float32,
        )
        regions[field_name] = Region(
            field=field_name,
            image=crop,
            conf=0.0,
            quad=quad,
            source="template",
        )
    return regions


def read_card(result, reader, image: np.ndarray | None = None, conf_min: float = 0.25) -> CardRead:
    bgr = image if image is not None else result.orig_img
    regions = fill_missing(bgr, regions_from_obb(result, bgr, conf_min=conf_min))
    out = CardRead(regions=regions)
    for field_name in ("name", "number", "collection"):
        region = regions[field_name]
        if field_name == "collection" and region.source == "template":
            continue
        text, ocr_conf, view = ocr_image(reader, region.image, field=field_name)
        region.image = view
        region.text = text
        region.ocr_conf = ocr_conf
        setattr(out, field_name, text)
    return out
