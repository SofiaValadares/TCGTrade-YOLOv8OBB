"""Roda o detector OBB e grava cada carta retificada (63 mm × 88 mm)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from .rectify import DEFAULT_DPI, card_size_px, crop_result
except ImportError:
    from rectify import DEFAULT_DPI, card_size_px, crop_result

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
_RUNS = REPO_ROOT / "OBB" / "runs" / "obb"
_V4_WEIGHTS = _RUNS / "obb-v4" / "weights" / "obb-v4.pt"
_V3_WEIGHTS = _RUNS / "obb-v3" / "weights" / "obb-v3.pt"
DEFAULT_OUTPUT = PACKAGE_DIR / "output"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def imgsz_for_weights(weights: Path) -> int:
    return 960 if "v4" in weights.name.lower() else 800


def find_default_weights() -> Path:
    """Pesos: env, pasta weights/ ao lado do código, ou treino em OBB/runs."""
    env = os.environ.get("TCG_CROPPER_WEIGHTS")
    if env:
        return Path(env)
    bundled = PACKAGE_DIR / "weights"
    for name in ("obb-v4.pt", "obb-v3.pt"):
        candidate = bundled / name
        if candidate.is_file():
            return candidate
    if bundled.is_dir():
        pts = sorted(bundled.glob("*.pt"))
        if pts:
            return pts[0]
    if _V4_WEIGHTS.is_file():
        return _V4_WEIGHTS
    return _V3_WEIGHTS


DEFAULT_WEIGHTS = find_default_weights()
DEFAULT_IMGSZ = imgsz_for_weights(DEFAULT_WEIGHTS)


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"Nenhuma imagem em {source}")
    return files


def save_crops(stem: str, crops, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for card in crops:
        base = f"{stem}_card_{card.index:02d}_{card.conf:.2f}"
        color_path = out_dir / f"{base}.jpg"
        bw_path = out_dir / f"{base}_bw.jpg"
        cv2.imwrite(str(color_path), card.image)
        cv2.imwrite(str(bw_path), card.image_bw)
        saved.append(color_path)
        saved.append(bw_path)
    return saved


def run(
    source: Path,
    weights: Path = DEFAULT_WEIGHTS,
    out_dir: Path = DEFAULT_OUTPUT,
    conf: float = 0.8,
    imgsz: int = DEFAULT_IMGSZ,
    dpi: int = DEFAULT_DPI,
    device: str | int = 0,
    refine: bool = False,
    inset: float = 0.02,
    enhance: bool = True,
    frame: bool = False,
    model: YOLO | None = None,
) -> list[Path]:
    if model is None:
        if not weights.exists():
            raise FileNotFoundError(
                f"Pesos OBB não encontrados: {weights}\n"
                "Passe --weights, defina TCG_CROPPER_WEIGHTS, "
                "exporte o CROPPER (python -m CROPPER export ...) "
                "ou treine em OBB/."
            )
        model = YOLO(str(weights))
    saved: list[Path] = []
    for image_path in collect_images(source):
        results = model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        crops = crop_result(
            results[0],
            dpi=dpi,
            refine=refine,
            inset=inset,
            enhance=enhance,
            frame=frame,
        )
        paths = save_crops(image_path.stem, crops, out_dir)
        saved.extend(paths)
        print(f"{image_path.name}: {len(crops)} carta(s) → {out_dir}")
    return saved


def parse_args() -> argparse.Namespace:
    width, height = card_size_px(DEFAULT_DPI)
    parser = argparse.ArgumentParser(
        description=(
            "Detecta cartas (YOLOv8 OBB) e retifica cada uma por perspectiva "
            f"para {width}×{height} px (63 mm × 88 mm @ {DEFAULT_DPI} DPI)."
        )
    )
    parser.add_argument("source", type=Path, help="Imagem ou pasta de imagens")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Snap antigo nas bordas (não usar para OCR).",
    )
    parser.add_argument(
        "--inset",
        type=float,
        default=0.02,
        help="Encolhe a caixa OBB em direcao ao centro (0.02 = 2 por cento).",
    )
    parser.add_argument("--no-enhance", action="store_true", help="Desliga equalizacao de luz.")
    parser.add_argument(
        "--frame",
        action="store_true",
        help="Encolhe a carta e pinta uma borda clara (desligado por padrao).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device: str | int = args.device
    if str(device).isdigit():
        device = int(device)
    run(
        source=args.source,
        weights=args.weights,
        out_dir=args.out,
        conf=args.conf,
        imgsz=args.imgsz,
        dpi=args.dpi,
        device=device,
        refine=args.refine,
        inset=args.inset,
        enhance=not args.no_enhance,
        frame=args.frame,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
