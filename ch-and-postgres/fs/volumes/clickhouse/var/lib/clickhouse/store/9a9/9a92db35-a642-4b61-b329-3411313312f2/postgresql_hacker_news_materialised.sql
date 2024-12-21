ATTACH TABLE _ UUID '4695cf80-5126-452f-9680-ecdde063d4d7'
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
    `descendants` Int64
)
ENGINE = MaterializedPostgreSQL('postgres:5432', 'clickhouse_pg_db', 'hacker_news', 'utopialvo', 'utopialvo')
ORDER BY (type, id)
