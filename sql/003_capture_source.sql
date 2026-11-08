-- Where a frame came from. 'camera' = live capture (firedrone.ingest), 'file_replay' = real image files uploaded from a
-- laptop folder (firedrone.upload_images, no Pi), 'synthetic' = generated test blobs (firedrone.synthetic).
-- Idempotent; safe on databases created with 001/002.
ALTER TABLE frames ADD COLUMN IF NOT EXISTS capture_source TEXT NOT NULL DEFAULT 'camera';
ALTER TABLE frames ADD COLUMN IF NOT EXISTS source_name TEXT;   -- e.g. original file name for file_replay
UPDATE frames SET capture_source = 'synthetic' WHERE is_synthetic AND capture_source = 'camera';
DO $$ BEGIN
    ALTER TABLE frames ADD CONSTRAINT frames_capture_source_chk
        CHECK (capture_source IN ('camera', 'file_replay', 'synthetic'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
