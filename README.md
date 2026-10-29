# Crossfake

Cross-dataset deepfake detection for **images and videos** (frame-level). Train on FaceForensics++ (c23), report **video-level AUC / EER** on FF++ test, Celeb-DF-v2, and DFDC Preview.

Videos are handled as uniformly sampled face crops; the video score is the **mean fake probability**.

## Methods

| Config | Backbone | Training |
| --- | --- | --- |
| `configs/baseline_efnb4.yaml` | EfficientNet-B4 (ImageNet) | Supervised real/fake on FF++ |
| `configs/sbi_efnb4.yaml` | EfficientNet-B4 | Self-Blended Images from **real** faces only |
| `configs/clip_probe.yaml` | Frozen CLIP ViT-B/16 + MLP | Linear probe on FF++ |

8 GB VRAM: mixed precision, 224×224, batch 8–16, gradient accumulation. Do not fine-tune CLIP-L.

Early stopping uses **intra FF++ validation video AUC**. Cross-dataset numbers come only from `src/eval.py`.

## Setup

```powershell
cd C:\Users\lenovo\Desktop\projects\crossfake
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If InsightFace GPU providers fail, install CPU ONNX: `pip install onnxruntime` and pass `--cpu` / `--cpu-faces` to extract/infer.

## Data

See [scripts/download.md](scripts/download.md). After the raw trees exist:

```powershell
python src/data/extract_faces.py --dataset ffpp --input data/raw/ffpp --compression c23 --num-frames 32
python src/data/extract_faces.py --dataset celebdf --input data/raw/celebdf --num-frames 32
python src/data/extract_faces.py --dataset dfdcp --input data/raw/dfdcp --num-frames 32
```

## Train / eval

```powershell
python src/train.py --config configs/baseline_efnb4.yaml
python src/eval.py --config configs/baseline_efnb4.yaml --checkpoint outputs/baseline_efnb4/best.pt

python src/train.py --config configs/sbi_efnb4.yaml
python src/eval.py --config configs/sbi_efnb4.yaml --checkpoint outputs/sbi_efnb4/best.pt

python src/train.py --config configs/clip_probe.yaml
python src/eval.py --config configs/clip_probe.yaml --checkpoint outputs/clip_probe/best.pt
```

Eval writes `cross_dataset_metrics.csv` and `.md` with frame- and video-level AUC/EER.

## Inference and demo

```powershell
python src/infer.py --config configs/baseline_efnb4.yaml --checkpoint outputs/baseline_efnb4/best.pt --input path\to\file.mp4
python app/gradio_app.py --config configs/baseline_efnb4.yaml --checkpoint outputs/baseline_efnb4/best.pt
python app/gradio_app.py --config configs/smoke.yaml --checkpoint outputs/smoke/best.pt --skip-face-detector
```

## Smoke test (no licensed data)

```powershell
python scripts/smoke_test.py
```

This writes synthetic crops, runs one EfficientNet-B0 epoch (`pretrained: false`), then eval + infer.
