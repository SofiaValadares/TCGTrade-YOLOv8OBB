# TCG Trade — Detecção e recorte de cartas

Dois estágios:

| Pasta | Função |
|---|---|
| [`OBB/`](OBB/) | YOLOv8 OBB: *existe uma carta aqui, nesta posição e neste ângulo?* |
| [`CROPPER/`](CROPPER/) | Perspectiva (scanner): recorte frontal **63 mm × 88 mm** |
| [`OCR/`](OCR/) | Faixas nome/número/coleção + EasyOCR |

A classe do detector de cartas é `card`. O nome do Pokémon é lido no [`OCR/`](OCR/).

```
TCGTradeSegmentacao/
  OBB/                 detector (notebook, docs, dataset local, runs)
  CROPPER/             recorte por perspectiva
  OCR/                 regiões de texto + leitura
  requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

1. Exporte o dataset YOLO OBB do Roboflow para `OBB/dataset/`.
2. Treine em [`OBB/train_yolov8_obb.ipynb`](OBB/train_yolov8_obb.ipynb) (`VERSION = "v4"`, fine-tune do v3).
3. Recorte no notebook [`CROPPER/crop_cards.ipynb`](CROPPER/crop_cards.ipynb) ou:

```powershell
.\.venv\Scripts\python.exe -m CROPPER OBB\dataset\test\images --out CROPPER\output\cards --conf 0.8
```

Para levar o recorte a outro projeto (código + pesos):

```powershell
.\.venv\Scripts\python.exe -m CROPPER export C:\outro-projeto\tcg_cropper
```

A saída é a carta inteira 63×88 mm (colorida + P&B). Leitura: notebook [`OCR/train_ocr.ipynb`](OCR/train_ocr.ipynb) (YOLO das faixas + EasyOCR).

`.venv/`, `OBB/dataset/`, `OBB/runs/`, `OCR/data/train|valid|test`, `OCR/runs/`, `CROPPER/output/` e pesos `.pt`/`.onnx` ficam fora do Git.
