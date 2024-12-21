ATTACH TABLE _ UUID '68d4c45a-45f2-4581-9ac6-dfdb0196491f'
(
    `title` String,
    `time` DateTime
)
ENGINE = MergeTree
ORDER BY time
SETTINGS index_granularity = 8192
