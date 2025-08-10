-- 最適化されたSQLクエリ
-- 元クエリID: 01f05bbf-565c-1be1-8932-f1f5ab3435dd
-- 最適化日時: 2025-08-10 07:47:37
-- ファイル: output_optimized_query_20250810-074737.sql

-- 🗂️ カタログ・スキーマ設定（自動追加）
USE CATALOG tpcds;
USE SCHEMA tpcds_sf1000_delta_lc;

WITH top_master_values AS (
SELECT
ID,
val
FROM master_itagaki
ORDER BY val DESC
LIMIT 10
)
SELECT
d.ID,
m.val,
d.cs_net_paid
FROM top_master_values m
JOIN detail_itagaki d
ON d.ID = m.ID
ORDER BY m.val DESC;