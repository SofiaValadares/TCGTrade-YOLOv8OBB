# TCG Trade — Detecção de Cartas com YOLOv8 OBB

Este projeto treina um **detector** de cartas de TCG (Pokémon) com **YOLOv8 Oriented Bounding Boxes (OBB)**.

Ele responde: *“existe uma carta aqui, nesta posição e neste ângulo?”*  
Ele **não** faz segmentação (máscara por pixel) e **não** reconhece o nome do Pokémon. A única classe é `card`.

O pipeline (validação do dataset, treino, métricas, gráficos e exportação ONNX) está em [`train_yolov8_obb.ipynb`](train_yolov8_obb.ipynb).

Comparações: [`docs/comparison-obb-v2-v3/README.md`](docs/comparison-obb-v2-v3/README.md) (atual) · [`docs/comparison-obb-v1-v2/README.md`](docs/comparison-obb-v1-v2/README.md) (histórico, dataset stretch).

## Dataset (local, fora do Git)

Anotações YOLO OBB: `class x1 y1 x2 y2 x3 y3 x4 y4`, projeto [Pokemon TCC](https://universe.roboflow.com/pokemon-tcc/pokemon-tcc) **v8**.

A pasta `dataset/` **não entra no Git**. Exporte **YOLOv8 OBB** do Roboflow para `dataset/` (`data.yaml`, `train/`, `valid/`, `test/`).

| Split | Imagens | Labels vazios (negativos) | Instâncias `card` |
|---|---|---|---|
| Treino | 245 | 19 | 1680 |
| Validação | 75 | 8 | 490 |
| Teste | 36 | 4 | 255 |
| **Total** | **356** | **31** | **2425** |

Pré-processamento: só auto-orientação EXIF. **Sem resize/stretch** e sem augmentation no export.

| Item | Recomendação |
|---|---|
| Auto-orient (EXIF) | Ligado |
| Resize | **Nenhum**. Se as fotos forem 4K, **Fit/Letterbox** no lado longo (ex. 1920), nunca Stretch |
| Augmentation no export | **Nenhuma** |
| Anotação | 4 cantos no **limite físico da carta** (borda impressa), não o plástico do binder |

## Como treinar (`obb-v3`)

No notebook, `VERSION = "v3"`. Checkpoint: `yolov8n-obb.pt`. Receita voltada a **caixa justa na borda**.

| Item | v2 | **v3** |
|---|---|---|
| Dataset | Roboflow v6, stretch 512 | **v8, sem resize** |
| `imgsz` | 640 | **800** |
| Box / DFL / angle | 7,5 / 1,5 / 1,0 | **10 / 2,0 / 1,5** |
| Rotação / scale / shear | 45° / 0,6 / 5 | **30° / 0,4 / 0** |
| Mixup | 0,05 | **0** |
| Mosaic desliga | últimas 15 épocas | **últimas 40** |
| Épocas | 120 | 120 (parou em **99**, early stop) |

Rode as células na ordem. Pesos: `runs/obb/obb-v3/weights/obb-v3.pt` e `.onnx`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Resultados do experimento `obb-v3`

Validação: **75 imagens / 490 cartas** (8 negativas). Teste: **36 imagens / 255 cartas** (4 negativas). Treino: 99 épocas (~27 min na RTX 5060), `imgsz=800`.

| Métrica | Validação | Teste | O que significa |
|---|---|---|---|
| **Recall** | 0.994 | **1.000** | Quase todas / todas as cartas anotadas foram encontradas. |
| **Precision** | 0.994 | 0.996 | Quase todas as caixas previstas eram carta de verdade. |
| **mAP@0.50** | 0.995 | 0.995 | A caixa orientada cobre a carta com IoU ≥ 50%. |
| **mAP@0.75** | 0.995 | 0.995 | Continua alto com encaixe mais apertado. |
| **mAP@0.50:0.95** | **0.994** | **0.994** | Localização + orientação (borda). Era 0.983 no v2 / ~0.93 quando o v2 é testado neste dataset nativo. |

Perdas na última época: box 0.185 / cls 0.123 / DFL 1.132 / angle 0.002. Angle ~0 = rotação colada.

Matriz do treino (val): **TP 489, FP 11, FN 1**. F1 da curva pico em conf **0,79** — no app, `conf=0.8` continua o limiar certo.

Na RTX 5060: ~4,4 ms de inferência + ~3,9 ms de pós-processamento por imagem (800 px).

![Curvas de treino e validação](docs/figures/results.png)

### Precision–Recall e F1

A curva PR no canto superior direito (mAP50 = 0.995). A F1 cai só depois de ~0,92 de confidence — não suba o `conf` para 0,9.

![Curva Precision–Recall](docs/figures/BoxPR_curve.png)

![Curva F1 vs confidence](docs/figures/BoxF1_curve.png)

### Matriz de confusão — como ler (importante)

O YOLO **não** classifica cada pixel como carta/fundo. Ele **casa caixas**:

|  | Verdade: carta | Verdade: fundo |
|---|---|---|
| Predito: carta | acerto (TP) — no v3: **489** | falso positivo (FP) — no v3: **11** |
| Predito: fundo | miss (FN) — no v3: **1** | **sempre 0** |

**Por que fundo→fundo é 0?** Não existem objetos “fundo” anotados. Fundo só aparece como *balde* de erros: predição sem carta (FP) ou carta sem predição (FN).

**Por que a normalizada parece modelo ruim?** Ela divide **cada coluna** pela soma da coluna. A coluna de fundo só tem os FPs, então FP/FP = **1.00**. Isso não significa que 100% da imagem foi classificada como carta.

Use a matriz de **contagens** + mAP. A normalizada, sozinha, engana.

![Matriz de confusão (contagens)](docs/figures/confusion_matrix.png)

![Matriz de confusão (normalizada)](docs/figures/confusion_matrix_normalized.png)

### Labels vs predições

Esquerda: anotações. Direita: o que o modelo desenhou. Conferir se as caixas colam na borda da carta.

![Validação — labels](docs/figures/val_batch0_labels.jpg)

![Validação — predições](docs/figures/val_batch0_pred.jpg)

## Inferência

```python
from ultralytics import YOLO

model = YOLO("runs/obb/obb-v3/weights/obb-v3.pt")
results = model.predict("dataset/test/images", conf=0.8, imgsz=800, save=True)
```

## Estrutura

```
dataset/                          # local (gitignore) — export Roboflow YOLO OBB
docs/figures/                     # gráficos do obb-v3 (versionados)
docs/comparison-obb-v1-v2/        # relatório v1 vs v2
docs/comparison-obb-v2-v3/        # relatório v2 vs v3
train_yolov8_obb.ipynb
requirements.txt
runs/obb/                         # treinos locais (não versionado)
```

`.venv/`, `dataset/`, `runs/` e pesos `.pt`/`.onnx` ficam fora do Git.
