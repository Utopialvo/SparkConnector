ATTACH TABLE _ UUID '1bb471ab-dd78-439f-b4c5-8c1eecb16459'
(
    `id` Int64,
    `text` String,
    `by` String
)
ENGINE = PostgreSQL('postgres:5432', 'clickhouse_pg_db', 'hacker_news', 'utopialvo', 'utopialvo')
