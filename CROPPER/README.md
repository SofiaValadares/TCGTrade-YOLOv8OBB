# CROPPER — carta frontal no tamanho de scanner

O CROPPER pega a caixa do detector OBB e devolve a **carta inteira**, de frente, no tamanho físico de uma carta Pokémon.

Ele **não treina** modelo e **não lê** o nome da carta. O treino fica em [`OBB/train_yolov8_obb.ipynb`](../OBB/train_yolov8_obb.ipynb). O OCR entra depois, em cima desta imagem.

Notebook: [`crop_cards.ipynb`](crop_cards.ipynb).

## O que sai

Cada recorte é uma imagem **63 mm × 88 mm** (retrato). Em 300 DPI: **744 × 1039 px**.

A carta ocupa a imagem inteira. Cada uma sai em **dois JPGs**:

| Arquivo | Uso |
|---|---|
| `<foto>_card_00_0.96.jpg` | colorida |
| `<foto>_card_00_0.96_bw.jpg` | preto e branco (OCR) |

O OCR pode usar sempre as mesmas regiões na carta **inteira** (colorida ou P&B):

| Campo | Onde na imagem (fração da altura × largura) |
|---|---|
| Nome | topo, ~5–13% da altura, lado esquerdo |
| HP | topo, lado direito |
| Número da set | rodapé, ~90–99% da altura, lado esquerdo |

Essas caixas estão em `OCR_TEMPLATE` (`enhance.py`). São só um guia — o arquivo gravado é a **carta completa**, não recortes de texto.

## Pipeline

```
foto  →  OBB (retângulo girado)  →  perspectiva 63×88 mm
      →  inset 2% (tira um pouco de plástico)
      →  luz mais plana + menos glare
      →  JPG colorido + JPG preto e branco
```

O OBB precisa estar na **borda impressa**. Se a anotação inclui o bolso ou a carta vizinha, o recorte herda isso — o CROPPER não “adivinha” canto na foto original.

## Uso

Na raiz deste repositório:

```powershell
.\.venv\Scripts\python.exe -m CROPPER caminho\foto.jpg --out CROPPER\output\cards --conf 0.8
.\.venv\Scripts\python.exe -m CROPPER OBB\dataset\test\images --out CROPPER\output\cards --conf 0.8
```

Arquivos: `<foto>_card_00_0.96.jpg` (cor) e `..._bw.jpg` (P&B).

## Exportar para outro projeto

Copia código + pesos `.pt` para uma pasta qualquer (não precisa deste repo depois):

```powershell
.\.venv\Scripts\python.exe -m CROPPER export C:\outro-projeto\tcg_cropper
```

No outro projeto:

```powershell
python C:\outro-projeto\tcg_cropper\crop_from_obb.py foto.jpg --out cartas
```

Ou instalar no venv de lá:

```powershell
python -m pip install -e C:\outro-projeto\tcg_cropper
tcg-crop foto.jpg --out cartas
```

Em Python:

```python
import sys
from pathlib import Path

sys.path.insert(0, r"C:\outro-projeto\tcg_cropper")
from crop_from_obb import run

run(source=Path(r"foto.jpg"), out_dir=Path(r"cartas"))
```

Os pesos vão em `tcg_cropper/weights/`. Alternativa: variável `TCG_CROPPER_WEIGHTS` ou `--weights`.

| Item | Padrão |
|---|---|
| Tamanho | 63×88 mm → 744×1039 px |
| Confidence | 0,8 |
| `imgsz` | 800 (`obb-v3`) ou 960 (`obb-v4`) |
| inset | 2% para o centro da caixa |
| enhance | luz / glare / nitidez leve |
| moldura | desligada (`--frame` pinta ~2% de borda clara) |

## Como o OCR deve usar

1. Abrir o JPG da carta inteira — **P&B** (`_bw.jpg`) costuma ler melhor; a colorida fica para conferência.
2. Recortar as regiões do `OCR_TEMPLATE` (ou a sua moldura).
3. Rodar Tesseract/EasyOCR **só nessas regiões**.

Não use faixas pré-cortadas pelo CROPPER — o layout da carta já é a moldura.

## Pastas

```
CROPPER/
  crop_from_obb.py    CLI
  export.py           copia código + pesos
  rectify.py          perspectiva 63×88
  enhance.py          luz, OCR_TEMPLATE
  pyproject.toml      pip install -e (tcg-crop)
  crop_cards.ipynb    visualização neste repo
  weights/            .pt no pacote exportado
  output/             gerado (gitignore)
```
