ATTACH TABLE _ UUID 'f6428fb1-7450-48ce-a07e-a58c1ea3e198'
(
    `id` Int64,
    `deleted` Int64,
    `type` String,
    `by` String,
    `time` DateTime,
    `text` String,
    `dead` Int64,
    `parent` Int64,
    `poll` Int64,
    `kids` Array(String),
    `url` String,
    `score` Int64,
    `title` String,
    `parts` Array(String),
    `descendants` Int64,
    `_sign` Int8 MATERIALIZED 1,
    `_version` UInt64 MATERIALIZED 1
)
ENGINE = ReplacingMergeTree(_version)
ORDER BY (type, id)
SETTINGS index_granularity = 8192
