ATTACH TABLE _ UUID 'ef1d02ab-b5fe-43c6-818f-4652a8858085'
(
    `id` Decimal(38, 19),
    `deleted` Nullable(Decimal(38, 19)),
    `type` String,
    `by` Nullable(String),
    `time` Nullable(DateTime64(6)),
    `text` Nullable(String),
    `dead` Nullable(Decimal(38, 19)),
    `parent` Nullable(Decimal(38, 19)),
    `poll` Nullable(Decimal(38, 19)),
    `kids` Array(Nullable(String)),
    `url` Nullable(String),
    `score` Nullable(Decimal(38, 19)),
    `title` Nullable(String),
    `parts` Array(Nullable(String)),
    `descendants` Nullable(Decimal(38, 19)),
    `_sign` Int8 MATERIALIZED 1,
    `_version` UInt64 MATERIALIZED 1
)
ENGINE = ReplacingMergeTree(_version)
ORDER BY (id, type)
SETTINGS index_granularity = 8192
