-- 001_seat_integrity.sql
--
-- Applied 2026-08-12 to the CardioX licence database (RDS `cardiox-staging`,
-- database `postgres`). These objects were created live while diagnosing
-- "Registration failed: No available seats for this license." and are recorded
-- here so the schema is reproducible — previously they existed only in the
-- running instance.
--
-- Background: the deployed `cardiox-license-register` Lambda has no hardware
-- fingerprint dedup. It INSERTs a new seat every time the same machine
-- registers, so one PC accumulated three seats on a two-seat licence and
-- exhausted it. Each failed cycle also drew a fresh licence from the pool and
-- stranded the previous one, permanently consuming keys.
--
-- Idempotent: safe to re-run.


-- ── Integrity constraints ────────────────────────────────────────────────────
-- One live seat per machine per licence. This is the structural guarantee that
-- duplicate seats cannot reappear even while the Lambda remains unfixed.
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS ux_seat_license_fp
    ON license_seats (license_id, bound_fingerprint)
    WHERE deleted_at IS NULL;

-- Seat numbers unique within a licence. Note the Lambda derives seat_number as
-- COUNT(*)+1, which collides after any row is deleted; this surfaces that bug
-- instead of letting it corrupt data silently.
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS ux_seat_license_num
    ON license_seats (license_id, seat_number)
    WHERE deleted_at IS NULL;

-- Lookup paths used on every register and heartbeat call.
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_seat_serial
    ON license_seats (rhythmulta_serial)
    WHERE status = 'active';

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_seat_fp
    ON license_seats (bound_fingerprint)
    WHERE deleted_at IS NULL;

-- `admin-create` only ever issues single/clinic/hospital/enterprise, but a
-- licence had been hand-inserted with plan_type 'multi' and max_seats 2, which
-- is what originally exhausted at three seats.
ALTER TABLE licenses DROP CONSTRAINT IF EXISTS ck_licenses_plan_type;
ALTER TABLE licenses ADD CONSTRAINT ck_licenses_plan_type
    CHECK (plan_type IN ('single', 'clinic', 'hospital', 'enterprise'));


-- ── TEMPORARY: dedup at the storage layer ────────────────────────────────────
-- REMOVE THIS once cardiox-license-register re-binds an existing seat instead
-- of inserting a new one.
--
-- ux_seat_license_fp rejects the Lambda's duplicate INSERT, and the Lambda has
-- no error handling, so the UniqueViolation reached the client as
-- "Internal server error". This trigger retires the machine's previous seat so
-- the incoming INSERT is the only live row, which keeps registration working
-- AND keeps repeat signups on the same licence instead of burning a new key.
--
-- It deliberately does NOT touch revoked rows: per the licensing rules a
-- revoked seat must keep blocking re-registration until an admin explicitly
-- releases it (released_at), so clearing one here would defeat revocation.
CREATE OR REPLACE FUNCTION supersede_duplicate_seat() RETURNS trigger AS $$
BEGIN
    UPDATE license_seats
       SET deleted_at = NOW(),
           status     = 'superseded'
     WHERE license_id        = NEW.license_id
       AND bound_fingerprint = NEW.bound_fingerprint
       AND deleted_at IS NULL
       AND status NOT IN ('revoked');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_supersede_duplicate_seat ON license_seats;
CREATE TRIGGER trg_supersede_duplicate_seat
    BEFORE INSERT ON license_seats
    FOR EACH ROW EXECUTE FUNCTION supersede_duplicate_seat();


-- ── Rollback ─────────────────────────────────────────────────────────────────
-- DROP TRIGGER IF EXISTS trg_supersede_duplicate_seat ON license_seats;
-- DROP FUNCTION IF EXISTS supersede_duplicate_seat();
-- ALTER TABLE licenses DROP CONSTRAINT IF EXISTS ck_licenses_plan_type;
-- DROP INDEX IF EXISTS ux_seat_license_fp;
-- DROP INDEX IF EXISTS ux_seat_license_num;
-- DROP INDEX IF EXISTS ix_seat_serial;
-- DROP INDEX IF EXISTS ix_seat_fp;


-- ── Operational note: releasing a seat ───────────────────────────────────────
-- Freeing a seat requires TWO statements. Updating license_seats alone leaves
-- the licence at status='active' with no seats, and because the register path
-- only draws licences with status='unused', that key is stranded permanently.
-- Three keys were lost this way before it was noticed.
--
--   UPDATE license_seats
--      SET status='revoked', revoked_at=NOW(), released_at=NOW()
--    WHERE rhythmulta_serial = :serial AND status='active';
--
--   UPDATE licenses l SET status='unused'
--    WHERE l.id = :license_id
--      AND NOT EXISTS (SELECT 1 FROM license_seats s
--                       WHERE s.license_id = l.id AND s.status = 'active');


-- ── Still outstanding in cardiox-license-register (not fixable in SQL) ───────
--   * `license_key` in the request body is ignored entirely — a key absent from
--     this database returns the same error as a valid one.
--   * No fingerprint dedup (the reason the trigger above exists).
--   * A revoked seat does not block re-registration, so a revoked user signs up
--     again within seconds. Needs: refuse when a seat exists for this
--     fingerprint with status='revoked' AND released_at IS NULL.
--   * seat_number is COUNT(*)+1 rather than MAX(seat_number)+1.
--   * Resolve/check/insert is not wrapped in a transaction, so two concurrent
--     signups can both pass the seat-cap check.
