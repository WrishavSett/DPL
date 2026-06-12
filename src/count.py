"""
RTSP Multi-Object Tracking Pipeline  (with line counting)

Stack : Python · OpenCV · Ultralytics (YOLO11n) · Supervision (ByteTrack)
Pipeline: Initialize stream > Load model > Init tracker > Init counters > Read frames > Preprocess > Infer > Track > Count (line) > Visualize

Usage:
    # RTSP stream
    python rtsp_mot_pipeline.py --source "rtsp://user:pass@<ip>:<port>/stream"

    # Local video file
    python rtsp_mot_pipeline.py --source /path/to/video.mp4

    # Webcam
    python rtsp_mot_pipeline.py --source 0

    # Cross-line counting  (format: x1,y1,x2,y2  in pixel coords)
    python rtsp_mot_pipeline.py --source 0 --count-line "0,240,640,240"

    # Save output to file
    python rtsp_mot_pipeline.py --source 0 --save

    # Run without display (headless)
    python rtsp_mot_pipeline.py --source 0 --headless

    # Save without display
    python rtsp_mot_pipeline.py --source 0 --save --headless
"""


#imports

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
CLASSES: list[str] | None = ["person"]  # e.g. ["person", "car"] — None means all classes

# Overlay colours (BGR)
COL_INFO    = (0, 255, 128)   # FPS / track count banner
COL_COUNT   = (0, 200, 255)   # per-class IN / OUT counts
COL_LINE    = (0, 255, 200)   # drawn counting line


# 1. Initialize Stream

def init_stream(source: str | int) ->cv2.VideoCapture:
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

def load_model(model_path: str, device: str) ->YOLO:
    """Load a YOLO model and move it to the target device."""
    print(f"[MODEL ] Loading '{model_path}' on {device.upper()} ...")
    model = YOLO(model_path)
    model.to(device)
    print(f"[MODEL ] Ready | classes: {len(model.names)}")
    return model


# 2b. Resolve class names

def resolve_classes(model: YOLO, classes: list[str] | None) ->list[int] | None:
    """Convert class-name strings to the integer IDs the model uses."""
    if classes is None:
        return None
    name_to_id = {v: k for k, v in model.names.items()}
    ids = []
    for name in classes:
        if name not in name_to_id:
            raise ValueError(
                f"Class '{name}' not found in model. "
                f"Available: {list(name_to_id)}"
            )
        ids.append(name_to_id[name])
    return ids


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


# 3b. Initialize Line Counter

def parse_line(spec: str | None) ->tuple[sv.Point, sv.Point] | None:
    """
    Parse  "x1,y1,x2,y2"  into a pair of sv.Point objects.
    Returns None when spec is None or empty.
    """
    if not spec:
        return None
    parts = [int(v) for v in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--count-line expects 'x1,y1,x2,y2', got: {spec!r}")
    return sv.Point(parts[0], parts[1]), sv.Point(parts[2], parts[3])


def init_line_counter(line_spec: str | None):
    """
    Build a LineZone from the CLI spec.

    Returns
    -------
    line_zone : sv.LineZone | None
    line_ann  : sv.LineZoneAnnotator | None
    """
    line_zone = line_ann = None

    points = parse_line(line_spec)
    if points is not None:
        start, end = points
        line_zone = sv.LineZone(start=start, end=end)
        line_ann  = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=0.6,
            color=sv.Color(r=COL_LINE[2], g=COL_LINE[1], b=COL_LINE[0]),
            text_color=sv.Color.WHITE,
            display_in_count=False,
            display_out_count=False,
        )
        print(f"[COUNT ] LineZone  > ({start.x},{start.y}) – ({end.x},{end.y})")

    return line_zone, line_ann


# 4 & 5. Read + Preprocess Frames

def read_frame(cap: cv2.VideoCapture) ->np.ndarray | None:
    """Grab the latest frame; return None on failure."""
    ret, frame = cap.read()
    return frame if ret else None


def preprocess(frame: np.ndarray, width: int, height: int) ->np.ndarray:
    """Resize frame to the inference resolution."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# 6. Inference

def run_inference(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    classes: list[int] | None
) ->sv.Detections:
    """Run YOLO inference and return a Supervision Detections object."""
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        classes=classes,
        verbose=True,
    )[0]
    return sv.Detections.from_ultralytics(results)


# 7. Object Tracking

def run_tracking(tracker: sv.ByteTrack, detections: sv.Detections) ->sv.Detections:
    """Update ByteTrack with current-frame detections."""
    return tracker.update_with_detections(detections)


# 7b. Line Counting

def run_counting(
    tracked:   sv.Detections,
    line_zone: sv.LineZone | None,
) ->dict:
    """
    Update the line counter and return a dict of results for the overlay.

    Keys returned
    -------------
    in_count_per_class  : dict[int, int]  – cumulative per-class IN counts
    out_count_per_class : dict[int, int]  – cumulative per-class OUT counts
    """
    counts = {"in_count_per_class": {}, "out_count_per_class": {}}

    if line_zone is not None:
        if len(tracked) >0:
            line_zone.trigger(detections=tracked)
        counts["in_count_per_class"]  = line_zone.in_count_per_class
        counts["out_count_per_class"] = line_zone.out_count_per_class

    return counts


# 8. Visualize

def build_labels(tracked: sv.Detections, model: YOLO) ->list[str]:
    """
    Format per-detection label: '#<tracker_id><class_name><conf>'.
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
    line_zone:  sv.LineZone | None,
    line_ann:   "sv.LineZoneAnnotator | None",
    counts:     dict,
    class_ids:  list[int] | None,
    class_names: dict[int, str],
) ->np.ndarray:
    """
    Draw bounding boxes, labels, line counter geometry, and an info overlay.
    """
    annotated = frame.copy()
    annotated = box_ann.annotate(annotated, tracked)
    annotated = label_ann.annotate(annotated, tracked, labels=labels)

    # Line zone
    if line_zone is not None and line_ann is not None:
        annotated = line_ann.annotate(frame=annotated, line_counter=line_zone)

    # Top-left HUD
    hud_lines = [
        (f"FPS: {fps:.1f}  |  Tracks: {len(tracked)}", COL_INFO),
    ]

    if line_zone is not None:
        in_per_class  = counts["in_count_per_class"]
        out_per_class = counts["out_count_per_class"]

        # Use configured class_ids if available, else fall back to seen classes
        display_ids = class_ids if class_ids is not None \
            else sorted(set(in_per_class) | set(out_per_class))

        for cls_id in display_ids:
            name = class_names.get(cls_id, str(cls_id))
            in_n  = in_per_class.get(cls_id, 0)
            out_n = out_per_class.get(cls_id, 0)
            hud_lines.append((f"IN {name}: {in_n}   OUT {name}: {out_n}", COL_COUNT))

    for row, (text, color) in enumerate(hud_lines):
        y = 24 + row * 28
        cv2.putText(
            annotated, text, (10, y),
            cv2.FONT_HERSHEY_COMPLEX, 0.7,
            color, 2, cv2.LINE_AA,
        )

    return annotated


# Helpers

def make_output_path(source: str | int) ->str:
    """Derive a sensible output filename from the source."""
    if isinstance(source, int):
        tag = f"webcam{source}"
    else:
        import os
        base = os.path.splitext(os.path.basename(source))[0]
        tag  = base if base else "stream"
    return f"output_{tag}.mp4"


def init_video_writer(path: str, frame: np.ndarray, fps: float) ->cv2.VideoWriter:
    """Create a VideoWriter sized to match *frame*."""
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"[SAVE  ] Failed to open VideoWriter at '{path}'")
    print(f"[SAVE  ] Writing > {path}  ({w}×{h} @ {fps:.1f} fps)")
    return writer


# Main Pipeline Loop

def main(
    source:    str | int,
    save:      bool,
    headless:  bool,
    line_spec: str | None,
) ->None:
    # Setup
    cap                         = init_stream(source)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 24.0
    print(f"[STREAM] Source FPS: {source_fps:.1f}")

    model                       = load_model(MODEL_PATH, DEVICE)
    class_ids                   = resolve_classes(model, CLASSES)
    tracker, box_ann, label_ann = init_tracker_and_annotators(
        conf_threshold=CONF_THRESHOLD,
        lost_track_buffer=LOST_TRACK_BUFFER,
    )

    # Counter setup
    line_zone, line_ann = init_line_counter(line_spec)

    fps_monitor    = sv.FPSMonitor()

    # Save setup
    writer      : cv2.VideoWriter | None = None
    output_path : str | None             = make_output_path(source) if save else None

    mode_tags = []
    if headless:  mode_tags.append("headless")
    if save:      mode_tags.append(f"saving > {output_path}")
    if line_zone: mode_tags.append("line counting")
    print(
        f"\n[PIPELINE] Running"
        f"{(' (' + ', '.join(mode_tags) + ')') if mode_tags else ''}"
        f" — {'Ctrl-C' if headless else 'press Q'} to quit\n"
    )

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
                model, processed, CONF_THRESHOLD, IOU_THRESHOLD, DEVICE, classes=class_ids
            )

            # 7. Tracking
            tracked = run_tracking(tracker, detections)

            # 7b. Counting
            counts = run_counting(tracked, line_zone)

            # 8. Visualize
            fps_monitor.tick()
            labels    = build_labels(tracked, model)
            annotated = annotate_frame(
                processed, tracked, labels, box_ann, label_ann,
                fps_monitor.fps,
                line_zone=line_zone,
                line_ann=line_ann,
                counts=counts,
                class_ids=class_ids,
                class_names=model.names,
            )

            # Save
            if save:
                if writer is None:
                    writer = init_video_writer(output_path, annotated, source_fps)
                writer.write(annotated)

            # Display
            if not headless:
                cv2.imshow("MOT Pipeline — Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[SAVE  ] Closed > {output_path}")
        if not headless:
            cv2.destroyAllWindows()

        # Final count summary
        print("\n[COUNT ] Final summary")
        if line_zone is not None:
            display_ids = class_ids if class_ids is not None \
                else sorted(set(line_zone.in_count_per_class) | set(line_zone.out_count_per_class))
            for cls_id in display_ids:
                name  = model.names.get(cls_id, str(cls_id))
                in_n  = line_zone.in_count_per_class.get(cls_id, 0)
                out_n = line_zone.out_count_per_class.get(cls_id, 0)
                print(f"[COUNT ]  {name:>12}  IN: {in_n}  OUT: {out_n}")
        else:
            print("[COUNT ]  No line counter configured.")
        print("[COUNT ]")

        print("[PIPELINE] Stopped.")


# Entry Point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RTSP + YOLO11n + ByteTrack MOT pipeline with line counting"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Stream source: RTSP URL, local video path, or webcam index (default: 0). "
            "Example: rtsp://admin:pass@192.168.1.100:554/stream"
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to a video file (output_<source>.mp4).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable live display window; run inference and tracking only.",
    )
    parser.add_argument(
        "--count-line",
        type=str,
        default=None,
        metavar="X1,Y1,X2,Y2",
        help=(
            "Draw a counting line between two points (in preprocessed frame pixels). "
            "Objects crossing left>right / top>bottom increment IN; the reverse increments OUT. "
            "Example: --count-line '0,240,640,240'"
        ),
    )

    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    main(
        source,
        save=args.save,
        headless=args.headless,
        line_spec=args.count_line,
    )