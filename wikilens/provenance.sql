CREATE TABLE IF NOT EXISTS provenance_run (
    report_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    capture_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    question TEXT NOT NULL,
    dataset_release VARCHAR(64) NOT NULL,
    capture_status VARCHAR(64) NOT NULL,
    captured_at VARCHAR(48) NOT NULL,
    runtime_json JSON NOT NULL,
    diagnostics_json JSON NOT NULL,
    capture_json LONGTEXT NOT NULL CHECK (JSON_VALID(capture_json)),
    imported_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;

CREATE TABLE IF NOT EXISTS provenance_evidence (
    report_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    evidence_index INT UNSIGNED NOT NULL,
    source_id VARCHAR(160) NOT NULL,
    lang CHAR(2) NOT NULL,
    page_id INT UNSIGNED NOT NULL,
    rev_id BIGINT UNSIGNED NOT NULL,
    wikidata_id VARCHAR(32),
    title VARCHAR(512) NOT NULL,
    revision_url VARCHAR(2048) NOT NULL,
    discovery_origin VARCHAR(64) NOT NULL,
    passage_start INT UNSIGNED NOT NULL,
    passage_end INT UNSIGNED NOT NULL,
    passage LONGTEXT NOT NULL,
    quotation TEXT NOT NULL,
    quote_start INT UNSIGNED NOT NULL,
    quote_end INT UNSIGNED NOT NULL,
    PRIMARY KEY (report_hash,evidence_index),
    KEY source_lookup (report_hash,source_id),
    CONSTRAINT provenance_evidence_run FOREIGN KEY (report_hash)
        REFERENCES provenance_run(report_hash)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;

CREATE TABLE IF NOT EXISTS provenance_claim (
    report_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    claim_index INT UNSIGNED NOT NULL,
    claim_text TEXT NOT NULL,
    PRIMARY KEY (report_hash,claim_index),
    CONSTRAINT provenance_claim_run FOREIGN KEY (report_hash)
        REFERENCES provenance_run(report_hash)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;

CREATE TABLE IF NOT EXISTS provenance_citation (
    report_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    claim_index INT UNSIGNED NOT NULL,
    evidence_index INT UNSIGNED NOT NULL,
    PRIMARY KEY (report_hash,claim_index,evidence_index),
    CONSTRAINT provenance_citation_claim FOREIGN KEY (report_hash,claim_index)
        REFERENCES provenance_claim(report_hash,claim_index),
    CONSTRAINT provenance_citation_evidence FOREIGN KEY (report_hash,evidence_index)
        REFERENCES provenance_evidence(report_hash,evidence_index)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;

CREATE TABLE IF NOT EXISTS provenance_review (
    report_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    reviewer VARCHAR(512) NOT NULL,
    reviewed_at VARCHAR(48) NOT NULL,
    review_json JSON NOT NULL,
    CONSTRAINT provenance_review_run FOREIGN KEY (report_hash)
        REFERENCES provenance_run(report_hash)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;

CREATE TABLE IF NOT EXISTS provenance_claim_review (
    report_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    claim_index INT UNSIGNED NOT NULL,
    supported BOOLEAN NOT NULL,
    source_ids_json JSON NOT NULL,
    PRIMARY KEY (report_hash,claim_index),
    CONSTRAINT provenance_claim_review_claim FOREIGN KEY (report_hash,claim_index)
        REFERENCES provenance_claim(report_hash,claim_index),
    CONSTRAINT provenance_claim_review_review FOREIGN KEY (report_hash)
        REFERENCES provenance_review(report_hash)
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
