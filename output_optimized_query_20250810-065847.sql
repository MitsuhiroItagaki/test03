-- 最適化されたSQLクエリ
-- 元クエリID: 01f05bbf-565c-1be1-8932-f1f5ab3435dd
-- 最適化日時: 2025-08-10 06:58:47
-- ファイル: output_optimized_query_20250810-065847.sql

-- 🗂️ カタログ・スキーマ設定（自動追加）
USE CATALOG tpcds;
USE SCHEMA tpcds_sf1000_delta_lc;

WITH filtered_master AS (
SELECT
ID,
val
FROM master_itagaki
WHERE ID IS NOT NULL
),
filtered_detail AS (
SELECT
ID,
cs_net_paid
FROM detail_itagaki
WHERE ID IS NOT NULL
),
joined_data AS (
SELECT
d.ID,
m.val,
d.cs_net_paid
FROM filtered_master m
JOIN filtered_detail d ON m.ID = d.ID
)
SELECT
ID,
val,
cs_net_paid
FROM joined_data
ORDER BY val DESC
LIMIT 10;