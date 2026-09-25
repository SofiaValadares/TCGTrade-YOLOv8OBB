"""Regiões de texto na carta recortada + OCR."""

from .read_card import CardRead, Region, read_card, regions_from_obb

__all__ = ["CardRead", "Region", "read_card", "regions_from_obb"]
