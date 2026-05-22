# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
YOLO Video Inference Script
Runs a trained Ultralytics YOLO model on a video file and saves annotated output.
"""

import sys
from pathlib import Path


# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------

def get_inputs():
    while True:
        video = input("Enter input video path: ").strip()
        video_path = Path(video).expanduser().resolve()
        if video_path.exists():
            break
        print("  [ERROR] Video not found: {}. Try again.".format(video_path))

    while True:
        model = input("Enter model weights path (.pt): ").strip()
        model_path = Path(model).expanduser().resolve()
        if model_path.exists():
            break
        print("  [ERROR] Model not found: {}. Try again.".format(model_path))

    default_output = video_path.parent / "{}_annotated.mp4".format(video_path.stem)
    output = input("Enter output video path [{}]: ".format(default_output)).strip()
    output_path = Path(output).expanduser().resolve() if output else default_output

    conf_raw = input("Confidence threshold [0.25]: ").strip()
    try:
        conf = float(conf_raw) if conf_raw else 0.25
    except ValueError:
        print("  [WARN] Invalid value, defaulting to 0.25")
        conf = 0.25

    return video_path, model_path, output_path, conf


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

def resolve_device():
    try:
        import torch
        if torch.cuda.is_available():
            print("  [OK] CUDA detected - using GPU 0")
            return "0"
    except ImportError:
        pass

    print("  [WARN] No CUDA detected - using CPU")
    return "cpu"


# ---------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------

def run_inference(video_path, model_path, output_path, conf, device):
    try:
        import cv2
    except ImportError:
        print("  [ERROR] opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Load model
    print("\n  Loading model: {}".format(model_path))
    model = YOLO(str(model_path))
    print("  [OK] Model loaded")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  [ERROR] Could not open video: {}".format(video_path))
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("  [OK] Video opened: {}x{}  {:.1f} fps  {} frames".format(width, height, fps, total))

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("  [ERROR] Could not open output writer: {}".format(output_path))
        sys.exit(1)

    print("  Output -> {}\n".format(output_path))

    # Inference loop
    frame_idx = 0
    log_every = max(1, total // 20)  # log ~20 times across the video

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, device=device, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % log_every == 0 or frame_idx == total:
            pct = frame_idx / total * 100 if total > 0 else 0
            print("  [{:5d}/{:5d}]  {:.1f}%".format(frame_idx, total, pct))

    cap.release()
    writer.release()

    print("\n  [OK] Done! Annotated video saved to: {}".format(output_path))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("\n+======================================+")
    print("|      YOLO Video Inference Script     |")
    print("+======================================+\n")

    video_path, model_path, output_path, conf = get_inputs()
    device = resolve_device()

    print("\n-- Summary --------------------------------")
    print("  Video  : {}".format(video_path))
    print("  Model  : {}".format(model_path))
    print("  Output : {}".format(output_path))
    print("  Conf   : {}".format(conf))
    print("  Device : {}".format(device))

    go = input("\n>> Start inference? [Y/n]: ").strip().lower()
    if go == "n":
        print("  Aborted.")
        return

    run_inference(video_path, model_path, output_path, conf, device)


if __name__ == "__main__":
    main()