-- Severity tracking over time. Continuous aggregate (incrementally refreshed) + latest-status view.
CREATE MATERIALIZED VIEW IF NOT EXISTS fire_severity_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '1 minute', captured_at)                           AS bucket,
    device_id,
    mission_id,
    count(*)                                                                AS n_frames,
    count(*) FILTER (WHERE status = 'ok')                                   AS n_confident,
    count(*) FILTER (WHERE status = 'uncertain')                            AS n_uncertain,
    count(*) FILTER (WHERE status = 'model_unavailable')                    AS n_no_model,
    count(*) FILTER (WHERE status = 'ok' AND fire_present)                  AS n_fire,
    avg(fire_probability)                                                   AS avg_fire_probability,
    max(severity_index) FILTER (WHERE status = 'ok' AND fire_present)       AS max_severity_index,
    avg(severity_index) FILTER (WHERE status = 'ok' AND fire_present)       AS avg_severity_index,
    avg(proxy_hot_pixel_fraction)                                           AS avg_proxy_hot_pixel_fraction,
    bool_or(is_synthetic)                                                   AS any_synthetic,
    bool_or(domain_mismatch)                                                AS any_domain_mismatch,
    bool_and(validated)                                                     AS all_validated
FROM fire_assessments
GROUP BY 1, 2, 3
WITH NO DATA;

SELECT add_continuous_aggregate_policy('fire_severity_1min',
    start_offset      => INTERVAL '1 day',
    end_offset        => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists     => true);

CREATE OR REPLACE VIEW fire_latest_by_device AS
SELECT DISTINCT ON (device_id) *
FROM fire_assessments
ORDER BY device_id, captured_at DESC, assessed_at DESC;
