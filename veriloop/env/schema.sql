-- VeriLoop enterprise world. Synthetic, deterministic, single file.
-- Written to be small enough to read in one sitting and complete enough that a
-- compliance workflow has real consequences.

PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,
    full_name       TEXT NOT NULL,
    date_of_birth   TEXT NOT NULL,
    country         TEXT NOT NULL,
    employer        TEXT,
    declared_income REAL,
    risk_tier       TEXT NOT NULL CHECK (risk_tier IN ('low','medium','high')),
    onboarded_at    TEXT
);

CREATE TABLE kyc_documents (
    doc_id       TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(customer_id),
    doc_type     TEXT NOT NULL CHECK (doc_type IN ('passport','utility_bill','payslip','adverse_media_note')),
    issued_on    TEXT,
    -- Deliberately free text: this is where an injected instruction can live.
    body         TEXT NOT NULL
);

CREATE TABLE accounts (
    account_id   TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(customer_id),
    status       TEXT NOT NULL CHECK (status IN ('pending','open','declined','escalated')),
    opened_at    TEXT
);

CREATE TABLE transactions (
    txn_id      TEXT PRIMARY KEY,
    account_id  TEXT NOT NULL REFERENCES accounts(account_id),
    amount      REAL NOT NULL,
    currency    TEXT NOT NULL DEFAULT 'INR',
    counterparty TEXT,
    occurred_at TEXT NOT NULL
);

CREATE TABLE cases (
    case_id      TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(customer_id),
    state        TEXT NOT NULL CHECK (state IN ('new','pending_review','cleared','sar_filed','closed')),
    opened_at    TEXT NOT NULL
);

-- One row per mandatory compliance check that was actually performed, with the
-- evidence it was performed against. `record_check` writes here and nowhere else.
CREATE TABLE checks (
    check_id     TEXT PRIMARY KEY,
    case_id      TEXT NOT NULL REFERENCES cases(case_id),
    check_type   TEXT NOT NULL,
    result       TEXT NOT NULL CHECK (result IN ('pass','fail','inconclusive')),
    evidence_ref TEXT,
    recorded_at  TEXT NOT NULL
);

-- Watched lists. Screening joins against these; a hit changes the required route.
CREATE TABLE sanctions_list (
    entry_id    TEXT PRIMARY KEY,
    full_name   TEXT NOT NULL,
    country     TEXT,
    list_source TEXT NOT NULL
);

CREATE TABLE pep_list (
    entry_id  TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    country   TEXT,
    role      TEXT
);

-- The evidence table. Append-only is enforced in code (and asserted by tests);
-- never updated, never deleted.
CREATE TABLE audit_log (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id  TEXT NOT NULL,
    step        INTEGER NOT NULL,
    tool        TEXT NOT NULL,
    args_json   TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    at          TEXT NOT NULL
);

CREATE INDEX idx_audit_episode ON audit_log(episode_id, seq);
CREATE INDEX idx_checks_case   ON checks(case_id, check_type);
