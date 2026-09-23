"""Recorte retificado de cartas a partir de OBB (63 mm × 88 mm)."""

from .enhance import OCR_TEMPLATE, apply_frame, draw_ocr_template, enhance_card, inset_quad, to_bw
from .rectify import (
    CARD_HEIGHT_MM,
    CARD_WIDTH_MM,
    DEFAULT_DPI,
    CroppedCard,
    card_size_px,
    crop_result,
    quads_from_obb_result,
    rectify_card,
)

__all__ = [
    "CARD_HEIGHT_MM",
    "CARD_WIDTH_MM",
    "DEFAULT_DPI",
    "OCR_TEMPLATE",
    "CroppedCard",
    "apply_frame",
    "card_size_px",
    "crop_result",
    "draw_ocr_template",
    "enhance_card",
    "inset_quad",
    "quads_from_obb_result",
    "rectify_card",
    "to_bw",
]
