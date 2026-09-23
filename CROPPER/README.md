# CROPPER — Recorte retificado de cartas

Segundo estágio: OBB → perspectiva **63 mm × 88 mm** (300 DPI = 744 × 1039 px) → tratamentos para **OCR** (nome, HP, número da set).

O CROPPER **não treina**. Treino do detector: [`OBB/train_yolov8_obb.ipynb`](../OBB/train_yolov8_obb.ipynb).

Notebook único: [`crop_cards.ipynb`](crop_cards.ipynb).

## Dois estágios

| Onde | O que faz |
|---|---|
| **OBB** | Caixa na **borda impressa** (anotação). Sem isso o recorte herda bolso/vizinha. |
| **CROPPER** | Warp + inset fixo (2%) + CLAHE/nitidez. **Não** procura borda na foto. |

O snap antigo (`--refine`) fica desligado: para OCR ele corta o nome ou deixa a carta do lado.

## Uso

```powershell
.\.venv\Scripts\python.exe CROPPER\crop_from_obb.py OBB\dataset\test\images --conf 0.8 --dpi 300
```

| Item | Padrão |
|---|---|
| Tamanho | 63×88 mm → 744×1039 px |
| `inset` | 0,02 (encolhe a caixa 2% ao centro) |
| enhance | CLAHE + nitidez |
| `imgsz` | 960 (v4) / 800 (v3) |
| `--refine` | off |

Faixas para o OCR (layout Pokémon, já retificado): topo = nome; rodapé = número da set (`ocr_bands`).
