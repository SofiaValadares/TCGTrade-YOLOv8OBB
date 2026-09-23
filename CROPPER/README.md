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

Na raiz do repositório, com o `.venv` e os pesos OBB:

```powershell
.\.venv\Scripts\python.exe CROPPER\crop_from_obb.py OBB\dataset\test\images --out CROPPER\output\cards --conf 0.8
```

Uma foto:

```powershell
.\.venv\Scripts\python.exe CROPPER\crop_from_obb.py caminho\foto.jpg --out CROPPER\output\cards
```

Arquivos: `CROPPER/output/cards/<foto>_card_00_0.96.jpg` (cor) e `..._bw.jpg` (P&B).

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
  rectify.py          perspectiva 63×88
  enhance.py          luz, moldura, OCR_TEMPLATE
  crop_cards.ipynb    visualização
  output/             gerado (gitignore)
```
