"""Database access (psycopg 3). Works with TigerData/Tiger Cloud or any PostgreSQL+TimescaleDB."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def connect(db_url: str, **kw) -> psycopg.Connection:
    if not db_url:
        raise RuntimeError("FIREDRONE_DB_URL is not set (TigerData: Console > service > Connect, use sslmode=require)")
    return psycopg.connect(db_url, row_factory=dict_row, **kw)


def apply_schema(conn: psycopg.Connection) -> None:
    for f in sorted(SQL_DIR.glob("*.sql")):
        with conn.cursor() as cur:
            cur.execute(f.read_text())
    conn.commit()


def upsert_device(conn, device_id: str, sensor_domain: str = "ir", camera_model: str | None = None,
                  description: str | None = None, radiometric: bool | None = None) -> None:
    conn.execute(
        """INSERT INTO devices (device_id, sensor_domain, camera_model, description, radiometric)
           VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (device_id) DO UPDATE SET sensor_domain = EXCLUDED.sensor_domain,
               camera_model = COALESCE(EXCLUDED.camera_model, devices.camera_model),
               description = COALESCE(EXCLUDED.description, devices.description),
               radiometric = EXCLUDED.radiometric""",
        (device_id, sensor_domain, camera_model, description, radiometric))
    conn.commit()


def insert_frame(conn, *, device_id: str, captured_at: datetime, image: bytes, width: int | None, height: int | None,
                 mission_id: str | None = None, sensor_domain: str = "ir", imu=None, is_synthetic: bool = False,
                 image_format: str = "jpeg", lat=None, lon=None, alt=None, capture_source: str | None = None,
                 source_name: str | None = None):
    """Insert one frame. Returns the frame_id, or None if this exact image was already stored (dedupe)."""
    sha = hashlib.sha256(image).hexdigest()
    g = (imu.gyro_x, imu.gyro_y, imu.gyro_z, imu.accel_x, imu.accel_y, imu.accel_z) if imu else (None,) * 6
    row = conn.execute(
        """INSERT INTO frames (captured_at, device_id, mission_id, image, image_format, width, height, sha256,
               sensor_domain, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, imu_ok,
               latitude, longitude, altitude_m, is_synthetic, capture_source, source_name)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (device_id, sha256, captured_at) DO NOTHING
           RETURNING frame_id""",
        (captured_at, device_id, mission_id, image, image_format, width, height, sha, sensor_domain,
         *g, imu is not None, lat, lon, alt, is_synthetic,
         capture_source or ("synthetic" if is_synthetic else "camera"), source_name)).fetchone()
    conn.commit()
    return row["frame_id"] if row else None


def claim_frames(conn, worker_id: str, limit: int, stale_after_s: int, max_attempts: int) -> list[dict]:
    """Atomically claim pending (or stale 'processing') frames. Safe with many concurrent workers."""
    rows = conn.execute(
        """WITH c AS (
               SELECT frame_id, captured_at FROM frames
               WHERE attempts < %(max)s AND (status = 'pending'
                     OR (status = 'processing' AND claimed_at < now() - make_interval(secs => %(stale)s)))
               ORDER BY captured_at
               LIMIT %(lim)s
               FOR UPDATE SKIP LOCKED)
           UPDATE frames f SET status = 'processing', claimed_at = now(), claimed_by = %(w)s,
                               attempts = f.attempts + 1
           FROM c WHERE f.frame_id = c.frame_id AND f.captured_at = c.captured_at
           RETURNING f.frame_id, f.captured_at, f.device_id, f.mission_id, f.image, f.sensor_domain,
                     f.gyro_x, f.gyro_y, f.gyro_z, f.is_synthetic, f.attempts""",
        {"max": max_attempts, "stale": stale_after_s, "lim": limit, "w": worker_id}).fetchall()
    conn.commit()
    return rows


def write_assessment(conn, frame: dict, a, *, model_name: str, model_version: str,
                     model_sha256: str | None, config_sha256: str) -> None:
    """Persist the output and mark the frame done in ONE transaction."""
    with conn.transaction():
        conn.execute(
            """INSERT INTO fire_assessments (captured_at, frame_id, device_id, mission_id, status, fire_present,
                   fire_probability, severity_class, severity_index, severity_confidence, class_probabilities,
                   proxy_hot_pixel_fraction, proxy_max_intensity, proxy_mean_intensity, quality_flags,
                   domain_mismatch, validated, model_name, model_version, model_sha256, config_sha256,
                   is_synthetic, notes)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (frame_id, captured_at, model_name, model_version) DO UPDATE SET
                   assessed_at = now(), status = EXCLUDED.status, fire_present = EXCLUDED.fire_present,
                   fire_probability = EXCLUDED.fire_probability, severity_class = EXCLUDED.severity_class,
                   severity_index = EXCLUDED.severity_index, severity_confidence = EXCLUDED.severity_confidence,
                   class_probabilities = EXCLUDED.class_probabilities, quality_flags = EXCLUDED.quality_flags,
                   notes = EXCLUDED.notes""",
            (frame["captured_at"], frame["frame_id"], frame["device_id"], frame["mission_id"], a.status,
             a.fire_present, a.fire_probability, a.severity_class, a.severity_index, a.severity_confidence,
             Jsonb(a.class_probabilities) if a.class_probabilities is not None else None,
             a.proxy_hot_pixel_fraction, a.proxy_max_intensity, a.proxy_mean_intensity, a.quality_flags,
             a.domain_mismatch, a.validated, model_name, model_version, model_sha256, config_sha256,
             frame["is_synthetic"], "; ".join(a.notes) or None))
        conn.execute("UPDATE frames SET status='done', last_error=NULL WHERE frame_id=%s AND captured_at=%s",
                     (frame["frame_id"], frame["captured_at"]))


def requeue(conn, mission_id: str | None = None, device_id: str | None = None) -> int:
    """Put already-processed frames back in the queue (e.g. after training a new model). Old outputs are kept;
    the new model writes new rows because outputs are unique per (frame, model_name, model_version)."""
    n = conn.execute(
        """UPDATE frames SET status = 'pending', attempts = 0, claimed_at = NULL, claimed_by = NULL, last_error = NULL
           WHERE status IN ('done', 'failed') AND (%(m)s::text IS NULL OR mission_id = %(m)s)
             AND (%(d)s::text IS NULL OR device_id = %(d)s)""", {"m": mission_id, "d": device_id}).rowcount
    conn.commit()
    return n


def fail_frame(conn, frame: dict, error: str, max_attempts: int) -> None:
    conn.execute(
        """UPDATE frames SET status = CASE WHEN attempts >= %s THEN 'failed' ELSE 'pending' END, last_error = %s
           WHERE frame_id = %s AND captured_at = %s""",
        (max_attempts, error[:2000], frame["frame_id"], frame["captured_at"]))
    conn.commit()
