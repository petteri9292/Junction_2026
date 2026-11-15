CREATE TABLE IF NOT EXISTS article (
    lang CHAR(2) NOT NULL,
    page_id INT UNSIGNED NOT NULL,
    rev_id BIGINT UNSIGNED NOT NULL,
    wikidata_id VARCHAR(32),
    title VARCHAR(512) NOT NULL,
    url VARCHAR(2048) NOT NULL,
    date_modified DATETIME,
    text MEDIUMTEXT NOT NULL,
    characters INT UNSIGNED NOT NULL,
    sections INT UNSIGNED NOT NULL,
    external_links INT UNSIGNED NOT NULL,
    PRIMARY KEY (lang, page_id),
    KEY topic (wikidata_id),
    FULLTEXT KEY article_keywords (title, text)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS article_vector (
    lang CHAR(2) NOT NULL,
    page_id INT UNSIGNED NOT NULL,
    model VARCHAR(64) NOT NULL,
    truncated BOOLEAN NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    PRIMARY KEY (lang, page_id),
    VECTOR INDEX article_semantics (embedding) M=16 DISTANCE=cosine
);
