"""Show what is in the database without needing psql (works the same on Windows/macOS/Linux).

    python -m firedrone.results                       # queue status + latest 20 assessments
    python -m firedrone.results --mission-id demo-1 --limit 50
    python -m firedrone.results --save-image <frame_id> --out frame.jpg
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import db


def queue_status(conn):
    return conn.execute("""SELECT capture_source, status, count(*) AS n FROM frames
                           GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()


def latest(conn, limit=20, mission_id=None, device_id=None):
    return conn.execute(
        """SELECT a.captured_at, a.device_id, a.mission_id, f.capture_source, f.source_name, a.status,
                  a.fire_present, a.fire_probability, a.severity_class, a.severity_confidence, a.validated,
                  a.domain_mismatch, a.model_name, a.frame_id, a.notes
           FROM fire_assessments a JOIN frames f USING (frame_id, captured_at)
           WHERE (%(m)s::text IS NULL OR a.mission_id = %(m)s) AND (%(d)s::text IS NULL OR a.device_id = %(d)s)
           ORDER BY a.captured_at DESC, a.assessed_at DESC LIMIT %(l)s""",
        {"m": mission_id, "d": device_id, "l": limit}).fetchall()


def save_image(conn, frame_id: str, out: Path) -> Path:
    r = conn.execute("SELECT image FROM frames WHERE frame_id = %s LIMIT 1", (frame_id,)).fetchone()
    if not r:
        raise SystemExit(f"frame {frame_id} not found")
    out.write_bytes(bytes(r["image"]))
    return out


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--mission-id")
    ap.add_argument("--device-id")
    ap.add_argument("--save-image", metavar="FRAME_ID")
    ap.add_argument("--out", type=Path, default=Path("frame.jpg"))
    ap.add_argument("--notes", action="store_true", help="also print the notes column")
    a = ap.parse_args(argv)
    conn = db.connect(os.environ.get("FIREDRONE_DB_URL"))
    if a.save_image:
        print("saved", save_image(conn, a.save_image, a.out))
        return
    print("== frames queue (capture_source / status / count)")
    for r in queue_status(conn):
        print(f"  {r['capture_source']:<12} {r['status']:<11} {r['n']}")
    rows = latest(conn, a.limit, a.mission_id, a.device_id)
    print(f"== latest {len(rows)} assessments (validated=False means the model has NOT passed held-out evaluation)")
    print(f"  {'captured_at (UTC)':<20} {'source':<28} {'status':<17} {'fire':<5} {'p_fire':<7} {'severity':<9} "
          f"{'sev_conf':<8} frame_id")
    for r in rows:
        src = (r["source_name"] or r["capture_source"])[:27]
        print(f"  {r['captured_at']:%Y-%m-%d %H:%M:%S} {src:<28} {r['status']:<17} {_fmt(r['fire_present']):<5} "
              f"{_fmt(r['fire_probability']):<7} {_fmt(r['severity_class']):<9} {_fmt(r['severity_confidence']):<8} "
              f"{r['frame_id']}")
        if a.notes and r["notes"]:
            print(f"      notes: {r['notes']}")


if __name__ == "__main__":
    main()
