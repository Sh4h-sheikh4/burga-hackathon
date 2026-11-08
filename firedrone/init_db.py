"""Check the database connection and create/upgrade the schema (idempotent; safe to re-run).

    python -m firedrone.init_db
Reads the connection string from FIREDRONE_DB_URL only.
"""
from __future__ import annotations

import os
import sys

from . import db


def main(argv=None):
    url = os.environ.get("FIREDRONE_DB_URL")
    if not url:
        sys.exit("FIREDRONE_DB_URL is not set. Set it to your TigerData service connection string (see HACKATHON_LAPTOP.md).")
    conn = db.connect(url, connect_timeout=15)
    ver = conn.execute("SELECT current_setting('server_version') v").fetchone()["v"]
    ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname='timescaledb'").fetchone()
    print(f"connected: PostgreSQL {ver}, timescaledb {ext['extversion'] if ext else 'not installed yet'}")
    if ext:
        major, minor = (int(x) for x in ext["extversion"].split(".")[:2])
        if (major, minor) < (2, 20):
            sys.exit("TimescaleDB >= 2.20 is required for the CREATE TABLE ... WITH (tsdb.hypertable) schema.")
    db.apply_schema(conn)
    hts = [r["hypertable_name"] for r in conn.execute(
        "SELECT hypertable_name FROM timescaledb_information.hypertables ORDER BY 1").fetchall()]
    print("schema ready; hypertables:", ", ".join(hts))


if __name__ == "__main__":
    main()
