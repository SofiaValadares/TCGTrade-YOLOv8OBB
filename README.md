# TCG Trade — Segmentação de Cartas com YOLOv8 OBB

Detector de cartas de TCG (Pokémon) com **YOLOv8 Oriented Bounding Boxes**. O modelo localiza cada carta mesmo quando ela está rotacionada na mesa, em vez de usar caixas alinhadas aos eixos.

O pipeline completo — validação do dataset, treino, métricas e exportação ONNX — está no notebook [`train_yolov8_obb.ipynb`](train_yolov8_obb.ipynb).

## Dataset

Anotações no formato YOLO OBB (`class x1 y1 x2 y2 x3 y3 x4 y4`), exportadas do Roboflow (projeto [Pokemon TCC](https://universe.roboflow.com/pokemon-tcc/pokemon-tcc), versão 5).

| Split | Imagens |
|---|---|
| Treino | 110 |
| Validação | 31 (236 instâncias) |
| Teste | 15 |
| **Total** | **156** |

- **Classe:** `card` (detecção genérica de carta, não do Pokémon impresso)
- **Pré-processamento:** auto-orientação EXIF e resize 512×512 (stretch)
- **Augmentation no Roboflow:** nenhuma

## Como o modelo foi treinado

Experimento **`obb-v1`**, a partir do checkpoint pré-treinado `yolov8n-obb.pt` (DOTAv1).

| Item | Valor |
|---|---|
| Arquitetura | YOLOv8n-OBB (3,07 M parâmetros, 8,3 GFLOPs) |
| Framework | Ultralytics 8.4.129 + PyTorch 2.11.0+cu128 |
| Hardware | NVIDIA GeForce RTX 5060 (8 GB) |
| Épocas | 100 |
| Tamanho de imagem | 640 |
| Batch | 16 |
| Early stopping | patience 20 |
| Seed | 42 |
| AMP | sim |
| Mosaic | ativo, desligado nas últimas 10 épocas |

Saída do treino: `runs/obb/obb-v1/`. Os pesos finais foram renomeados e exportados:

- PyTorch: `runs/obb/obb-v1/weights/obb-v1.pt` (6,4 MB)
- ONNX: `runs/obb/obb-v1/weights/obb-v1.onnx` (11,9 MB)

## Resultados (`obb-v1`)

Validação no split `valid` (31 imagens, 236 cartas), via `model.val()`:

| Métrica | Valor |
|---|---|
| mAP@0.50 | **0.994** |
| mAP@0.75 | **0.994** |
| mAP@0.50:0.95 | **0.985** |
| Precision | **0.983** |
| Recall | **0.996** |

Por classe (`card`): Precision 0.983, Recall 0.996, AP50 0.994, AP50-95 0.985.

Perdas na última época de treino (`results.csv`):

| Perda | Valor |
|---|---|
| Box Loss | 0.212 |
| Class Loss | 0.243 |
| DFL Loss | 1.194 |
| Angle Loss | 0.015 |

Inferência na RTX 5060: ~4,6 ms por imagem (640×640), mais ~5,9 ms de pós-processamento.

Há uma única classe e o conjunto de validação é pequeno, então os números medem localização/orientação da carta — não identificação do nome do Pokémon.

Gráficos gerados no experimento: `confusion_matrix.png`, `BoxF1_curve.png`, `BoxPR_curve.png`, `results.png` e lotes `val_batch*_labels.jpg` / `val_batch*_pred.jpg`.

## Como treinar de novo

1. Python 3.14+ e GPU NVIDIA (RTX 50-series precisa de PyTorch **cu128**).
2. Crie a venv e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Abra `train_yolov8_obb.ipynb` com o kernel da `.venv`.
4. Ajuste `VERSION` (ex.: `"v2"`) e execute as células na ordem.

Para um dataset novo, substitua a pasta `dataset/` por um export YOLOv8 OBB do Roboflow (com `data.yaml`) e rode de novo a célula de validação do dataset.

## Inferência

```python
from ultralytics import YOLO

model = YOLO("runs/obb/obb-v1/weights/obb-v1.pt")
results = model.predict("dataset/test/images", save=True)
```

## Estrutura

```
dataset/                 # imagens + labels OBB + data.yaml
train_yolov8_obb.ipynb   # treino, validação, métricas e export
requirements.txt
runs/obb/obb-v1/         # artefatos do experimento (não versionado)
```

Pesos (`.pt`/`.onnx`), `runs/` e `.venv/` ficam de fora do Git.
