from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import build_transforms  # noqa: E402
from src.data.extract_faces import crop_face, init_face_app, uniform_frame_indices  # noqa: E402
from src.engine import autocast_ctx  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import get_device, load_checkpoint, load_config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_model(config_path: str, checkpoint_path: str, device=None):
    cfg = load_config(config_path)
    device = device or get_device()
    ckpt = load_checkpoint(checkpoint_path, device)
    model_cfg = dict(cfg)
    if "config" in ckpt:
        model_cfg["model"] = ckpt["config"]["model"]
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tf = build_transforms(int(cfg["data"]["image_size"]), train=False)
    return cfg, model, tf, device


def load_detector(config_path: str, checkpoint_path: str, device=None, cpu_faces: bool = False):
    cfg, model, tf, device = load_model(config_path, checkpoint_path, device=device)
    app = init_face_app(ctx_id=-1 if cpu_faces or device.type == "cpu" else 0)
    return cfg, model, app, tf, device


def _to_tensor(rgb: np.ndarray, tf) -> torch.Tensor:
    out = tf(image=rgb)["image"]
    return torch.from_numpy(out).permute(2, 0, 1).float()


@torch.no_grad()
def score_crops(model, crops_rgb: list[np.ndarray], tf, device, use_amp: bool = True) -> dict:
    if not crops_rgb:
        return {"score": float("nan"), "n_faces": 0, "frame_scores": []}
    batch = torch.stack([_to_tensor(c, tf) for c in crops_rgb]).to(device)
    with autocast_ctx(device, use_amp):
        logits = model(batch)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits).flatten().float().cpu().numpy()
    score = float(np.mean(probs))
    return {"score": score, "n_faces": int(len(probs)), "frame_scores": probs.tolist()}


def infer_image(path: Path, model, app, tf, device, margin: float, use_amp: bool) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    result = crop_face(bgr, app, margin=margin)
    if result is None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = score_crops(model, [rgb], tf, device, use_amp)
        out["note"] = "no face detected; scored full image"
        return out
    crop, _ = result
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return score_crops(model, [rgb], tf, device, use_amp)


def infer_video(path: Path, model, app, tf, device, margin: float, num_frames: int, use_amp: bool) -> dict:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = uniform_frame_indices(n, num_frames)
    crops = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        result = crop_face(frame, app, margin=margin)
        if result is None:
            continue
        crop, _ = result
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    cap.release()
    return score_crops(model, crops, tf, device, use_amp)


def infer_raw(path: Path, model, tf, device, cfg, use_amp: bool) -> dict:
    """Score without a face detector (full frames / pre-cropped faces)."""
    ext = path.suffix.lower()
    num_frames = int(cfg.get("eval", {}).get("frames_per_video", 32))
    if ext in IMAGE_EXTS:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = score_crops(model, [rgb], tf, device, use_amp)
        result["note"] = "scored without face detector"
        return result, "image"
    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        crops = []
        for idx in uniform_frame_indices(n, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                crops.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        result = score_crops(model, crops, tf, device, use_amp)
        result["note"] = "scored video frames without face detector"
        return result, "video"
    raise ValueError(f"Unsupported file type: {ext}")


def infer_path(path: Path, model, app, tf, device, cfg, use_amp: bool = True) -> dict:
    margin = float(cfg["data"].get("margin", 0.3))
    num_frames = int(cfg.get("eval", {}).get("frames_per_video", 32))
    ext = path.suffix.lower()
    if app is None:
        result, kind = infer_raw(path, model, tf, device, cfg, use_amp)
    elif ext in IMAGE_EXTS:
        kind = "image"
        result = infer_image(path, model, app, tf, device, margin, use_amp)
    elif ext in VIDEO_EXTS:
        kind = "video"
        result = infer_video(path, model, app, tf, device, margin, num_frames, use_amp)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    score = result["score"]
    result.update(
        {
            "path": str(path),
            "media": kind,
            "label": "fake" if score >= 0.5 else "real",
            "fake_probability": score,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score an image or video (mean face probability).")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--cpu-faces", action="store_true")
    p.add_argument(
        "--assume-cropped",
        action="store_true",
        help="Treat the input image as an already cropped face (skip RetinaFace).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    if args.assume_cropped:
        cfg, model, tf, device = load_model(args.config, args.checkpoint)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = score_crops(model, [rgb], tf, device, use_amp=bool(cfg["train"].get("amp", True)))
        score = result["score"]
        result.update(
            {
                "path": str(path),
                "media": "image",
                "label": "fake" if score >= 0.5 else "real",
                "fake_probability": score,
                "note": "assumed pre-cropped face",
            }
        )
    else:
        cfg, model, app, tf, device = load_detector(args.config, args.checkpoint, cpu_faces=args.cpu_faces)
        result = infer_path(path, model, app, tf, device, cfg, use_amp=bool(cfg["train"].get("amp", True)))
    print(result)


if __name__ == "__main__":
    main()
