# OCR — regiões na carta + leitura

O CROPPER entrega a carta inteira (63×88 mm, P&B). Daqui saem **duas coisas**:

1. **Locais de interesse** — YOLOv8 OBB com 3 classes na carta já recortada: `name`, `number`, `colection`.
2. **Leitura** — EasyOCR só nessas faixas (não treina OCR do zero).

Notebook: [`train_ocr.ipynb`](train_ocr.ipynb).

## Dataset (local, fora do Git)

Export **YOLOv8 OBB** do Roboflow para `OCR/data/` (`data.yaml`, `train/`, `valid/`, `test/`).

Imagens = cartas do CROPPER em P&B. Labels = 4 cantos da faixa de texto (`class x1 y1 … x4 y4`).

| Split | Imagens |
|---|---|
| Treino | 205 |
| Validação | 59 |
| Teste | 29 |
| **Total** | **293** cartas · **292** name · **292** number · **190** colection |

A classe `colection` (typo do Roboflow) **não está em toda carta** — só quando o nome da set está escrito. Não renomeie no `data.yaml`.

## Como treinar / testar

Na raiz do repo ou em `OCR/`:

1. Abra [`train_ocr.ipynb`](train_ocr.ipynb).
2. `VERSION = "v1"` → pesos em `OCR/runs/ocr-roi-v1/weights/ocr-roi-v1.pt`.
3. Rode as células: dataset → treino YOLO → galeria das caixas → EasyOCR nas faixas.

O OCR usa as caixas do detector; se faltar `name`/`number`, cai na moldura fixa. `collection` sem caixa = vazio (a maioria das cartas).

## Pastas

```
OCR/
  train_ocr.ipynb    treino + teste (regiões e OCR)
  read_card.py       recorte da OBB + EasyOCR
  data/              Roboflow (gitignore nas imagens)
  runs/              treino (gitignore)
```
