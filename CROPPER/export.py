"""Copia o CROPPER + pesos .pt para outra pasta (outro projeto, outro PC)."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CODE_FILES = (
    "crop_from_obb.py",
    "rectify.py",
    "enhance.py",
    "__init__.py",
    "__main__.py",
    "export.py",
    "pyproject.toml",
    "README.md",
)
REQUIREMENTS = "ultralytics\nopencv-python\nnumpy\n"


def find_weights_to_copy(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env = os.environ.get("TCG_CROPPER_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    bundled = PACKAGE_DIR / "weights"
    for name in ("obb-v4.pt", "obb-v3.pt"):
        candidate = bundled / name
        if candidate.is_file():
            return candidate
    if bundled.is_dir():
        pts = sorted(bundled.glob("*.pt"))
        if pts:
            return pts[0]
    runs = REPO_ROOT / "OBB" / "runs" / "obb"
    for rel in ("obb-v4/weights/obb-v4.pt", "obb-v3/weights/obb-v3.pt"):
        candidate = runs / rel
        if candidate.is_file():
            return candidate
    return runs / "obb-v3" / "weights" / "obb-v3.pt"


def export_cropper(dest: Path, weights: Path | None = None) -> Path:
    dest = dest.expanduser().resolve()
    src_root = PACKAGE_DIR.resolve()
    if dest == src_root:
        raise ValueError("O destino não pode ser a pasta CROPPER deste repositório.")
    src_weights = find_weights_to_copy(weights)
    if not src_weights.is_file():
        raise FileNotFoundError(
            f"Pesos não encontrados: {src_weights}\n"
            "Treine o OBB ou passe --weights caminho\\obb-v3.pt"
        )

    dest.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        src = src_root / name
        if not src.is_file():
            raise FileNotFoundError(f"Falta arquivo do CROPPER: {src}")
        shutil.copy2(src, dest / name)

    (dest / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    weights_dir = dest / "weights"
    weights_dir.mkdir(exist_ok=True)
    shutil.copy2(src_weights, weights_dir / src_weights.name)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exporta o CROPPER (código + pesos) para usar fora deste repositório.",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="Pasta de destino (criada se nao existir).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Pesos .pt (padrao: os mesmos que o CROPPER usa aqui).",
    )
    args = parser.parse_args()
    dest = export_cropper(args.dest, weights=args.weights)
    weights_path = next((dest / "weights").glob("*.pt"))
    script = dest / "crop_from_obb.py"
    print(f"Exportado: {dest}")
    print(f"Pesos    : {weights_path}")
    print()
    print("No outro projeto:")
    print(f"  python {script} foto.jpg --out cartas")
    print("ou:")
    print(f"  python -m pip install -e {dest}")
    print("  tcg-crop foto.jpg --out cartas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
