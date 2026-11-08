import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

TEST_DSN = os.environ.get("FIREDRONE_TEST_DSN")


@pytest.fixture
def conn():
    if not TEST_DSN:
        pytest.skip("FIREDRONE_TEST_DSN not set (needs PostgreSQL + TimescaleDB)")
    from firedrone import db
    c = db.connect(TEST_DSN)
    with c.cursor() as cur:
        cur.execute("DROP MATERIALIZED VIEW IF EXISTS fire_severity_1min CASCADE")
        cur.execute("DROP VIEW IF EXISTS fire_latest_by_device")
        cur.execute("DROP TABLE IF EXISTS fire_assessments, frames, devices CASCADE")
    c.commit()
    db.apply_schema(c)
    yield c
    c.close()
