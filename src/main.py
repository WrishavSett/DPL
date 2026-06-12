"""
RTSP Multi-Object Tracking Pipeline

Stack : Python · OpenCV · Ultralytics (YOLO11n) · Supervision (ByteTrack)
Pipeline: Initialize stream > Load model > Init tracker > Read frames > Throttle > Preprocess > Infer > Track > Visualize

Usage:
    # RTSP stream
    python rtsp_mot_pipeline.py --source "rtsp://user:pass@<ip>:<port>/stream"

    # Local video file
    python rtsp_mot_pipeline.py --source /path/to/video.mp4

    # Webcam
    python rtsp_mot_pipeline.py --source 0
"""


# imports

import argparse
import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# Configuration

MODEL_PATH         = "yolo26n.pt"       # any Ultralytics-compatible weight
TARGET_W, TARGET_H = 640, 480           # preprocessing resolution
CONF_THRESHOLD     = 0.5
IOU_THRESHOLD      = 0.5
DEVICE             = "cpu"              # "cpu" or "cuda"
LOST_TRACK_BUFFER  = 30                 # frames to keep a lost track alive


# 1. Initialize Stream

def init_stream(source: str | int) -> cv2.VideoCapture:
    """
    Open an RTSP stream, local video file, or webcam.
    For RTSP sources, forces the FFmpeg backend and minimises buffer size so we always decode the latest frame.
    """
    print(f"[STREAM] Connecting > {source}")

    is_rtsp = isinstance(source, str) and source.startswith("rtsp")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if is_rtsp \
        else cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[STREAM] Connected")
    return cap


# 2. Load Model

def load_model(model_path: str, device: str) -> YOLO:
    """Load a YOLO model and move it to the target device."""
    print(f"[MODEL ] Loading '{model_path}' on {device.upper()} ...")
    model = YOLO(model_path)
    model.to(device)
    print(f"[MODEL ] Ready | classes: {len(model.names)}")
    return model

# 3. Initialize ByteTrack Tracker + Supervision Annotators

def init_tracker_and_annotators(
    conf_threshold: float,
    lost_track_buffer: int,
):
    """
    ByteTrack via Supervision.
    Annotators (instantiated once at startup, reused every frame):
      - BoxAnnotator   : draws detection boxes
      - LabelAnnotator : prints tracker-ID + class name + confidence
    """
    tracker = sv.ByteTrack(
        track_activation_threshold=conf_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=0.8,
    )

    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    print("[TRACK ] ByteTrack initialised")
    return tracker, box_annotator, label_annotator


# 4 & 5. Read + Preprocess Frames

def read_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Grab the latest frame; return None on failure."""
    ret, frame = cap.read()
    return frame if ret else None


def preprocess(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to the inference resolution."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# 6. Inference

def run_inference(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
) -> sv.Detections:
    """Run YOLO inference and return a Supervision Detections object."""
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False,
    )[0]
    return sv.Detections.from_ultralytics(results)


# 7. Object Tracking

def run_tracking(tracker: sv.ByteTrack, detections: sv.Detections) -> sv.Detections:
    """Update ByteTrack with current-frame detections."""
    return tracker.update_with_detections(detections)


# 8. Visualize

def build_labels(tracked: sv.Detections, model: YOLO) -> list[str]:
    """
    Format per-detection label: '#<tracker_id> <class_name> <conf>'.
    Guards against None fields that can occur when there are no detections.
    """
    labels = []
    for i in range(len(tracked)):
        cls_id   = int(tracked.class_id[i])    if tracked.class_id   is not None else -1
        track_id = int(tracked.tracker_id[i])  if tracked.tracker_id is not None else -1
        conf     = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
        cls_name = model.names.get(cls_id, str(cls_id))
        labels.append(f"#{track_id} {cls_name} {conf:.2f}")
    return labels


def annotate_frame(
    frame:      np.ndarray,
    tracked:    sv.Detections,
    labels:     list[str],
    box_ann:    sv.BoxAnnotator,
    label_ann:  sv.LabelAnnotator,
    fps:        float,
) -> np.ndarray:
    """Draw bounding boxes, labels, and a track-count/FPS overlay."""
    annotated = frame.copy()
    annotated = box_ann.annotate(annotated, tracked)
    annotated = label_ann.annotate(annotated, tracked, labels=labels)

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}  |  Tracks: {len(tracked)}",
        (10, 24),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (0, 255, 128),
        2,
        cv2.LINE_AA,
    )
    return annotated


# Main Pipeline Loop

def main(source: str | int) -> None:
    # Setup
    cap                         = init_stream(source)
    model                       = load_model(MODEL_PATH, DEVICE)
    tracker, box_ann, label_ann = init_tracker_and_annotators(
        conf_threshold=CONF_THRESHOLD,
        lost_track_buffer=LOST_TRACK_BUFFER,
    )

    fps_monitor = sv.FPSMonitor()

    print("\n[PIPELINE] Running — press Q to quit\n")

    try:
        while True:
            # 4. Read frame
            frame = read_frame(cap)
            if frame is None:
                print("[STREAM] Frame read failed — attempting reconnect ...")
                time.sleep(1)
                cap.release()
                cap = init_stream(source)
                continue

            # 5. Preprocess
            processed = preprocess(frame, TARGET_W, TARGET_H)

            # 6. Inference
            detections = run_inference(
                model, processed, CONF_THRESHOLD, IOU_THRESHOLD, DEVICE
            )

            # 7. Tracking
            tracked = run_tracking(tracker, detections)

            # 8. Visualize
            fps_monitor.tick()
            labels    = build_labels(tracked, model)
            annotated = annotate_frame(processed, tracked, labels, box_ann, label_ann, fps_monitor.fps)

            cv2.imshow("MOT Pipeline — Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[PIPELINE] Stopped.")


# Entry Point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTSP + YOLO11n + ByteTrack MOT pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Stream source: RTSP URL, local video path, or webcam index (default: 0). "
            "Example: rtsp://admin:pass@192.168.1.100:554/stream"
        ),
    )
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    main(source)