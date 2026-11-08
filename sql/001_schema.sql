-- Fire-drone schema for TigerData (Tiger Cloud) / self-hosted TimescaleDB >= 2.23.
-- Syntax per https://docs.tigerdata.com/api/latest/hypertable/create_table/ (CREATE TABLE ... WITH (tsdb.hypertable ...)).
-- Validated locally on PostgreSQL 17.11 + TimescaleDB 2.23.0 (built from source). Idempotent: safe to re-run.
-- Self-hosted TimescaleDB 2.20-2.22: tsdb.partition_column is set explicitly below, so it also works there
--   (add a columnstore policy manually if you want compression). <= 2.19: use create_hypertable() instead.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Plain relational table: one row per drone / camera rig.
CREATE TABLE IF NOT EXISTS devices (
    device_id      TEXT PRIMARY KEY,
    description    TEXT,
    camera_model   TEXT,
    sensor_domain  TEXT NOT NULL DEFAULT 'ir' CHECK (sensor_domain IN ('ir', 'rgb', 'unknown')),
    -- 'ir' does not say WHICH IR (near-IR vs thermal); record the exact camera in camera_model.
    radiometric    BOOLEAN,                          -- NULL = unknown; TRUE only if confirmed calibrated temperature output
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Raw camera frames + IMU snapshot. Doubles as the inference work queue (status column).
-- Columnstore is OFF because the worker updates status on recent rows and rows hold large bytea blobs.
CREATE TABLE IF NOT EXISTS frames (
    captured_at    TIMESTAMPTZ NOT NULL,
    frame_id       UUID        NOT NULL DEFAULT gen_random_uuid(),
    device_id      TEXT        NOT NULL REFERENCES devices(device_id),
    mission_id     TEXT,
    image          BYTEA       NOT NULL,
    image_format   TEXT        NOT NULL DEFAULT 'jpeg',
    width          INTEGER,
    height         INTEGER,
    sha256         TEXT        NOT NULL,
    sensor_domain  TEXT        NOT NULL DEFAULT 'ir' CHECK (sensor_domain IN ('ir', 'rgb', 'unknown')),
    -- LSM6DS3TR-C snapshot (NULL when IMU unavailable). gyro in deg/s, accel in g.
    gyro_x DOUBLE PRECISION, gyro_y DOUBLE PRECISION, gyro_z DOUBLE PRECISION,
    accel_x DOUBLE PRECISION, accel_y DOUBLE PRECISION, accel_z DOUBLE PRECISION,
    imu_ok         BOOLEAN NOT NULL DEFAULT FALSE,
    latitude DOUBLE PRECISION, longitude DOUBLE PRECISION, altitude_m DOUBLE PRECISION,
    is_synthetic   BOOLEAN NOT NULL DEFAULT FALSE,   -- TRUE for generated test fixtures, never real camera data
    status         TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    attempts       INTEGER NOT NULL DEFAULT 0,
    claimed_at     TIMESTAMPTZ,
    claimed_by     TEXT,
    last_error     TEXT,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (frame_id, captured_at)
) WITH (
    tsdb.hypertable,
    tsdb.partition_column = 'captured_at',
    tsdb.chunk_interval = '1 day',
    tsdb.columnstore = false
);
CREATE UNIQUE INDEX IF NOT EXISTS frames_dedupe ON frames (device_id, sha256, captured_at);
CREATE INDEX IF NOT EXISTS frames_queue ON frames (status, captured_at) WHERE status IN ('pending', 'processing');

-- Model outputs ("output type" fire_assessment_v1). One row per frame per model version.
-- No FK to frames: TimescaleDB forbids hypertable->hypertable foreign keys; (frame_id, captured_at) joins back.
CREATE TABLE IF NOT EXISTS fire_assessments (
    captured_at          TIMESTAMPTZ NOT NULL,
    frame_id             UUID        NOT NULL,
    device_id            TEXT        NOT NULL,
    mission_id           TEXT,
    assessed_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    output_type          TEXT        NOT NULL DEFAULT 'fire_assessment_v1',
    status               TEXT        NOT NULL CHECK (status IN ('ok', 'uncertain', 'model_unavailable', 'error')),
    fire_present         BOOLEAN,             -- NULL = unknown (uncertain / no model). Never defaulted to FALSE.
    fire_probability     REAL,
    severity_class       TEXT,                -- model's severity label; NULL unless status='ok' and fire_present
    severity_index       SMALLINT,            -- ordinal of severity_class (0 = lowest)
    severity_confidence  REAL,
    class_probabilities  JSONB,
    -- Relative image-brightness statistics. NOT temperature, NOT hazard. Camera type unknown (near-IR or thermal);
    -- many cameras use auto gain/exposure, so values may not be comparable across frames.
    proxy_hot_pixel_fraction REAL,
    proxy_max_intensity      REAL,
    proxy_mean_intensity     REAL,
    quality_flags        TEXT[] NOT NULL DEFAULT '{}',
    domain_mismatch      BOOLEAN NOT NULL DEFAULT FALSE,  -- model training domain != camera domain
    validated            BOOLEAN NOT NULL DEFAULT FALSE,  -- TRUE only once model has passed held-out evaluation
    model_name           TEXT NOT NULL,
    model_version        TEXT NOT NULL,
    model_sha256         TEXT,
    config_sha256        TEXT,
    is_synthetic         BOOLEAN NOT NULL DEFAULT FALSE,
    notes                TEXT,
    UNIQUE (frame_id, captured_at, model_name, model_version)
) WITH (
    tsdb.hypertable,
    tsdb.partition_column = 'captured_at',
    tsdb.chunk_interval = '1 day',
    tsdb.segmentby = 'device_id',
    tsdb.orderby = 'captured_at DESC'
);
CREATE INDEX IF NOT EXISTS fire_assessments_mission ON fire_assessments (mission_id, captured_at DESC);
