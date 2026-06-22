-- Schema for the multi-camera counting system.
-- WAL mode and busy_timeout are set programmatically per-connection in
-- app/database.py (not here) since they're connection pragmas, not schema.

-- One row per configured camera. This table is the source of truth for
-- Tab 2 (management) and drives what CameraManager spawns on startup.
CREATE TABLE IF NOT EXISTS cameras (
    camera_id          TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    source             TEXT NOT NULL,                  -- RTSP URL / file path / webcam index
    enabled            INTEGER NOT NULL DEFAULT 1,      -- 0/1, user-controlled
    classes            TEXT,                            -- JSON array of class names; NULL = all classes
    count_line         TEXT,                            -- JSON "[x1,y1,x2,y2]"; NULL = no line counter
    model_path         TEXT,                            -- NULL = fall back to app default
    device             TEXT    NOT NULL DEFAULT 'cpu',
    target_w           INTEGER NOT NULL DEFAULT 640,
    target_h           INTEGER NOT NULL DEFAULT 480,
    conf_threshold     REAL    NOT NULL DEFAULT 0.5,
    iou_threshold       REAL    NOT NULL DEFAULT 0.5,
    lost_track_buffer  INTEGER NOT NULL DEFAULT 30,
    status             TEXT    NOT NULL DEFAULT 'stopped',  -- stopped|starting|running|crashed|disabled
    pid                INTEGER,
    last_heartbeat     TEXT,                            -- ISO8601 UTC, updated by worker heartbeat
    created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Append-only log of individual line-crossing events. This is the ground
-- truth for all counts; cumulative/hourly figures are derived by querying
-- this table, never by reading a worker's in-memory counter. Surviving
-- camera restarts and crashes falls out of this for free.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT    NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    track_id    INTEGER NOT NULL,
    class_id    INTEGER NOT NULL,
    class_name  TEXT    NOT NULL,
    direction   TEXT    NOT NULL CHECK (direction IN ('in', 'out')),
    timestamp   TEXT    NOT NULL                        -- ISO8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events (camera_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_time        ON events (timestamp);

-- Index of generated hourly CSV reports, so Tab 4 can list/download without
-- re-scanning the reports/ directory on every request.
CREATE TABLE IF NOT EXISTS reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL UNIQUE,
    period_start  TEXT NOT NULL,                        -- ISO8601 UTC, inclusive
    period_end    TEXT NOT NULL,                         -- ISO8601 UTC, exclusive
    generated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);