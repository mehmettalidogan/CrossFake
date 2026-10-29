from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer import infer_path, load_detector, load_model  # noqa: E402

BUNDLE = {}


def _as_path(file_obj) -> Path | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, dict):
        p = file_obj.get("path") or file_obj.get("name")
        return Path(p) if p else None
    if hasattr(file_obj, "name"):
        return Path(file_obj.name)
    return Path(file_obj)


def predict(file_obj):
    if not BUNDLE:
        return "Load a checkpoint first.", 0.0
    path = _as_path(file_obj)
    if path is None:
        return "Upload an image or video.", 0.0
    result = infer_path(
        path,
        BUNDLE["model"],
        BUNDLE["app"],
        BUNDLE["tf"],
        BUNDLE["device"],
        BUNDLE["cfg"],
        use_amp=True,
    )
    score = result.get("fake_probability", float("nan"))
    if result.get("n_faces", 0) == 0:
        text = "No face found; cannot score."
        return text, 0.0
    label = result["label"]
    n = result["n_faces"]
    media = result["media"]
    text = f"{media}: {label} (fake probability {score:.3f}, {n} face crop(s))"
    if result.get("note"):
        text += f"\n{result['note']}"
    return text, float(score)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Crossfake") as demo:
        gr.Markdown(
            """
            # Crossfake
            Frame-level deepfake detector. Videos are scored by **mean fake probability** over sampled face crops.
            """
        )
        inp = gr.File(label="Image or video", file_types=["image", "video"])
        out_text = gr.Textbox(label="Result")
        out_score = gr.Number(label="Fake probability")
        btn = gr.Button("Score")
        btn.click(predict, inputs=inp, outputs=[out_text, out_score])
    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline_efnb4.yaml")
    p.add_argument("--checkpoint", default="outputs/baseline_efnb4/best.pt")
    p.add_argument("--cpu-faces", action="store_true")
    p.add_argument(
        "--skip-face-detector",
        action="store_true",
        help="Do not load RetinaFace; score the uploaded image/frames as-is.",
    )
    p.add_argument("--share", action="store_true")
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7860)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_face_detector:
        cfg, model, tf, device = load_model(args.config, args.checkpoint)
        app = None
    else:
        cfg, model, app, tf, device = load_detector(args.config, args.checkpoint, cpu_faces=args.cpu_faces)
    BUNDLE.update({"cfg": cfg, "model": model, "app": app, "tf": tf, "device": device})
    demo = build_ui()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
