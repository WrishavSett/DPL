"""
Turn a single frame's line-crossing results into persistable event records.

CameraWorker calls extract_events() once per frame; any CountEvent objects
returned are put on the shared multiprocessing.Queue that app/db_writer.py
drains into the `events` table. This is the only place "a crossing
happened" gets translated into data — worker/pipeline.py only exposes the
raw boolean masks from LineZone.trigger().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import supervision as sv

from worker.pipeline import CountingResult


@dataclass
class CountEvent:
    """One track crossing the line, in one direction, at one moment."""

    camera_id: str
    track_id: int
    class_id: int
    class_name: str
    direction: str    # "in" | "out"
    timestamp: str     # ISO8601 UTC

    def to_dict(self) -> dict:
        return asdict(self)


# def _utc_now_iso() -> str:    # Old
#     return datetime.now(timezone.utc).isoformat()
def _utc_now_iso() -> str:      # New
    # Fixed-width microseconds so the stored TEXT timestamps sort/compare
    # correctly as plain strings — datetime.isoformat() omits microseconds
    # when they happen to be exactly zero, which would otherwise corrupt
    # range queries like the hourly report's WHERE timestamp >= ? AND < ?.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def extract_events(
    camera_id: str,
    tracked: sv.Detections,
    result: CountingResult,
    class_names: dict[int, str],
) -> list[CountEvent]:
    """
    Build a CountEvent for every detection that crossed the line this frame.

    `result.crossed_in` / `result.crossed_out` are boolean arrays aligned
    index-for-index with `tracked`, exactly as returned by
    LineZone.trigger() — nothing here is re-derived from the cumulative
    (and restart-resettable) in_count_per_class / out_count_per_class dicts.
    """
    events: list[CountEvent] = []

    if result.crossed_in is None and result.crossed_out is None:
        return events

    now = _utc_now_iso()
    n = len(tracked)

    for i in range(n):
        track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
        class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
        class_name = class_names.get(class_id, str(class_id))

        if result.crossed_in is not None and bool(result.crossed_in[i]):
            events.append(
                CountEvent(
                    camera_id=camera_id,
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    direction="in",
                    timestamp=now,
                )
            )

        if result.crossed_out is not None and bool(result.crossed_out[i]):
            events.append(
                CountEvent(
                    camera_id=camera_id,
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    direction="out",
                    timestamp=now,
                )
            )

    return events