-- ============================================================
-- Today in History — Supabase schema
--
-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor)
-- to create the table and supporting objects.
-- ============================================================

-- 1. Main table
CREATE TABLE IF NOT EXISTS historical_events (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    year        INTEGER     NOT NULL,
    month       INTEGER     NOT NULL CHECK (month BETWEEN 1 AND 12),
    day         INTEGER     NOT NULL CHECK (day BETWEEN 1 AND 31),
    title       TEXT        NOT NULL,
    description TEXT        NOT NULL,
    wikipedia_url TEXT      DEFAULT '',
    ai_summary  TEXT        DEFAULT '',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now(),

    -- Prevent duplicate events for the same date + title.
    CONSTRAINT uq_event UNIQUE (year, month, day, title)
);

-- 2. Index for fast date lookups (the primary query pattern).
CREATE INDEX IF NOT EXISTS idx_events_date
    ON historical_events (month, day, year);

-- 3. Auto-update `updated_at` on every row change.
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_updated_at
    BEFORE UPDATE ON historical_events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- 4. Enable Row Level Security (Supabase best practice).
ALTER TABLE historical_events ENABLE ROW LEVEL SECURITY;

-- 5. Allow public read access (anon key).
CREATE POLICY "Allow public read"
    ON historical_events FOR SELECT
    USING (true);

-- 6. Allow service-role inserts/updates (used by the backend).
CREATE POLICY "Allow service insert/update"
    ON historical_events FOR ALL
    USING (true)
    WITH CHECK (true);
