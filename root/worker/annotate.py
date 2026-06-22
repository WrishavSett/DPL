"""
Drawing the per-frame HUD (camera_id + classwise counts) and box/label
overlays, then JPEG-encoding the result for the Tab 3 visualization feed.

Adapted from count.py's build_labels/annotate_frame: the box/label drawing
and FPS banner are kept, but the top-left HUD now leads with the camera_id
(per the Tab 3 requirement: "camera_id and classwise_counts displayed on
the top left") and counts come from worker.pipeline.CountingResult instead
of a loosely-typed dict.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from worker.pipeline import CountingResult

# Overlay colours (BGR)
COL_CAMERA = (255, 255, 255)   # camera_id banner
COL_INFO   = (0, 255, 128)     # FPS / track count
COL_COUNT  = (0, 200, 255)     # per-class IN / OUT counts


def build_labels(tracked: sv.Detections, class_names: dict[int, str]) -> list[str]:
    """Format per-detection label: '#<tracker_id> <class_name> <conf>'."""
    labels = []
    for i in range(len(tracked)):
        track_id = tracked.tracker_id[i] if tracked.tracker_id is not None else -1
        class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
        conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
        cls_name = class_names.get(class_id, str(class_id))
        labels.append(f"#{track_id} {cls_name} {conf:.2f}")
    return labels


def annotate_frame(
    frame: np.ndarray,
    camera_id: str,
    tracked: sv.Detections,
    labels: list[str],
    box_ann: sv.BoxAnnotator,
    label_ann: sv.LabelAnnotator,
    fps: float,
    line_zone: sv.LineZone | None,
    line_ann: "sv.LineZoneAnnotator | None",
    counting_result: CountingResult,
    class_ids: list[int] | None,
    class_names: dict[int, str],
) -> np.ndarray:
    """
    Draw bounding boxes, labels, line counter geometry, and a top-left HUD
    that leads with camera_id + classwise counts, per the Tab 3 spec.
    """
    annotated = frame.copy()
    annotated = box_ann.annotate(annotated, tracked)
    annotated = label_ann.annotate(annotated, tracked, labels=labels)

    if line_zone is not None and line_ann is not None:
        annotated = line_ann.annotate(frame=annotated, line_counter=line_zone)

    hud_lines: list[tuple[str, tuple[int, int, int]]] = [
        (camera_id, COL_CAMERA),
        (f"FPS: {fps:.1f}  |  Tracks: {len(tracked)}", COL_INFO),
    ]

    if line_zone is not None:
        in_per_class = counting_result.in_count_per_class
        out_per_class = counting_result.out_count_per_class
        display_ids = class_ids if class_ids is not None \
            else sorted(set(in_per_class) | set(out_per_class))

        for cls_id in display_ids:
            name = class_names.get(cls_id, str(cls_id))
            in_n = in_per_class.get(cls_id, 0)
            out_n = out_per_class.get(cls_id, 0)
            hud_lines.append((f"{name}: IN {in_n}  OUT {out_n}", COL_COUNT))

    for row, (text, color) in enumerate(hud_lines):
        y = 24 + row * 26
        cv2.putText(
            annotated, text, (10, y),
            cv2.FONT_HERSHEY_COMPLEX, 0.6,
            color, 2, cv2.LINE_AA,
        )

    return annotated


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a frame to JPEG bytes for publishing to shared state / streaming."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buffer.tobytes()