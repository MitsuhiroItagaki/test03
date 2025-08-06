# Databricks SQL Performance Optimization Report

## 1. Executive Summary

The query execution time of 20.6 seconds is acceptable, but several optimization opportunities exist:

| Metric | Current Status | Evaluation | Priority |
|--------|---------------|------------|----------|
| Execution Time | 20.6s | ✅ Good | - |
| Data Read Volume | 33.95GB | ⚠️ High | Medium |
| Photon Enabled | Yes | ✅ Good | - |
| Shuffle Operations | 58 times | ⚠️ High | Medium |
| Cache Efficiency | 53.7% | ⚠️ Low | Medium |
| Filter Efficiency | 90.2% | ✅ Good | - |
| Data Skew | AQE Handled | ✅ Good | - |
| Spill Occurrence | None | ✅ Good | - |

## 2. Time-Consuming Operations Analysis

### Top 10 Operations by Execution Time

1. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.store_sales)**
   - Time: 99.9s (14.7% of total)
   - Memory: 3043.2 MB
   - Tasks: 189
   - **Current clustering key**: ss_sold_date_sk, ss_item_sk
   - **Filter rate**: 98.0% (read: 159.60GB, pruned: 156.41GB)

2. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.store_sales)**
   - Time: 67.7s (9.9% of total)
   - Memory: 3051.3 MB
   - Tasks: 189
   - **Current clustering key**: ss_sold_date_sk, ss_item_sk
   - **Filter rate**: 98.0% (read: 159.60GB, pruned: 156.43GB)

3. **Photon Grouping Aggregate**
   - Time: 62.1s (9.1% of total)
   - Memory: 797.6 MB
   - Tasks: 189

4. **Photon Inner Join**
   - Time: 58.6s (8.6% of total)
   - Memory: 242.0 MB
   - Tasks: 189

5. **Photon Grouping Aggregate**
   - Time: 36.9s (5.4% of total)
   - Memory: 799.0 MB
   - Tasks: 189

6. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.catalog_sales)**
   - Time: 30.6s (4.5% of total)
   - Memory: 3048.1 MB
   - Tasks: 189
   - **Current clustering key**: cs_item_sk, cs_sold_date_sk
   - **Filter rate**: 98.7% (read: 121.20GB, pruned: 119.60GB)

7. **Photon Aggregate**
   - Time: 29.0s (4.3% of total)
   - Memory: 0.0 MB
   - Tasks: 643

8. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.catalog_sales)**
   - Time: 25.6s (3.8% of total)
   - Memory: 3049.5 MB
   - Tasks: 189
   - **Current clustering key**: cs_item_sk, cs_sold_date_sk
   - **Filter rate**: 99.1% (read: 121.20GB, pruned: 120.15GB)

9. **Photon Inner Join**
   - Time: 23.2s (3.4% of total)
   - Memory: 242.0 MB
   - Tasks: 189

10. **Photon Left Semi Join**
    - Time: 22.1s (3.2% of total)
    - Memory: 418.0 MB
    - Tasks: 189

## 3. Liquid Clustering Analysis

### High-Priority Tables

#### 1. tpcds.tpcds_sf1000_delta_lc.store_sales
- **Table Size**: 159.60GB
- **Current clustering key**: ss_sold_date_sk, ss_item_sk
- **Recommended clustering key**: ss_sold_date_sk, ss_item_sk
- **Filter rate**: 90.21% (high efficiency)
- **Assessment**: Current clustering key is optimal - no changes needed

#### 2. tpcds.tpcds_sf1000_delta_lc.catalog_sales
- **Table Size**: 121.20GB
- **Current clustering key**: cs_item_sk, cs_sold_date_sk
- **Recommended clustering key**: cs_item_sk, cs_sold_date_sk
- **Assessment**: Current clustering key is optimal - no changes needed

#### 3. tpcds.tpcds_sf1000_delta_lc.web_sales
- **Table Size**: 60.17GB
- **Current clustering key**: ws_item_sk, ws_sold_date_sk
- **Recommended clustering key**: ws_item_sk, ws_sold_date_sk
- **Assessment**: Current clustering key is optimal - no changes needed

### Low-Priority Tables

#### 4. tpcds.tpcds_sf1000_delta_lc.item
- **Table Size**: 0.03GB
- **Current clustering key**: i_item_sk
- **Recommendation**: Use CACHE TABLE instead of clustering due to small size
- **Implementation**:
  ```sql
  CACHE TABLE tpcds.tpcds_sf1000_delta_lc.item;
  ```

#### 5. tpcds.tpcds_sf1000_delta_lc.date_dim
- **Table Size**: 0.00GB
- **Current clustering key**: d_date_sk, d_year
- **Recommendation**: Use CACHE TABLE instead of clustering due to small size
- **Implementation**:
  ```sql
  CACHE TABLE tpcds.tpcds_sf1000_delta_lc.date_dim;
  ```

## 4. Query Optimization Analysis

### Optimization Results

- **Optimization trials**: Multiple attempts executed
- **Selected solution**: Query with structural optimizations (JOIN reordering, predicate pushdown)
- **Optimization note**: While the core query logic and business requirements remain unchanged, several execution plan optimizations were successfully applied to improve performance
- **Applied optimizations**:
  - JOIN order restructuring (small tables prioritized)
  - Early filter condition application for better data pruning
  - Predicate pushdown optimization for reduced data movement
  - Subquery optimization and common table expression restructuring
- **Cost reduction**: 32.0% (based on EXPLAIN COST comparison between original and optimized execution plans)
- **Memory efficiency improvement**: 0% (no memory-specific optimizations applied)
- **JOIN operations reduction**: 88 → 69 operations (21.6% reduction through query restructuring)

### Applied Optimization Techniques

1. JOIN order optimization (small table first)
2. Early filter condition application
3. Predicate pushdown optimization

### Optimized SQL Query

```sql
USE CATALOG tpcds;
USE SCHEMA tpcds_sf1000_delta_lc;
WITH date_ranges AS (
SELECT
d_date_sk,
d_year,
d_week_seq,
CASE
WHEN d_year = 1998 AND d_moy = 12 AND d_dom = 19 THEN 'last_year_week'
WHEN d_year = 1999 AND d_moy = 12 AND d_dom = 19 THEN 'this_year_week'
END AS week_type
FROM date_dim
WHERE (d_year = 1998 AND d_moy = 12 AND d_dom = 19) OR
(d_year = 1999 AND d_moy = 12 AND d_dom = 19) OR
(d_year BETWEEN 1998 AND 2002)
),
target_weeks AS (
SELECT
CASE WHEN week_type = 'last_year_week' THEN d_week_seq END AS last_year_week,
CASE WHEN week_type = 'this_year_week' THEN d_week_seq END AS this_year_week
FROM date_ranges
WHERE week_type IS NOT NULL
),
common_items AS (
SELECT
i.i_item_sk,
i.i_brand_id,
i.i_class_id,
i.i_category_id
FROM item i
WHERE EXISTS (
SELECT 1 FROM store_sales ss
JOIN date_ranges d ON ss.ss_sold_date_sk = d.d_date_sk AND d.d_year BETWEEN 1998 AND 2002
WHERE ss.ss_item_sk = i.i_item_sk
)
AND EXISTS (
SELECT 1 FROM catalog_sales cs
JOIN date_ranges d ON cs.cs_sold_date_sk = d.d_date_sk AND d.d_year BETWEEN 1998 AND 2002
WHERE cs.cs_item_sk = i.i_item_sk
)
AND EXISTS (
SELECT 1 FROM web_sales ws
JOIN date_ranges d ON ws.ws_sold_date_sk = d.d_date_sk AND d.d_year BETWEEN 1998 AND 2002
WHERE ws.ws_item_sk = i.i_item_sk
)
),
avg_sales AS (
SELECT AVG(sales_value) AS average_sales
FROM (
SELECT ss_quantity * ss_list_price AS sales_value
FROM store_sales ss
JOIN date_ranges d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE d.d_year BETWEEN 1998 AND 2002
UNION ALL
SELECT cs_quantity * cs_list_price AS sales_value
FROM catalog_sales cs
JOIN date_ranges d ON cs.cs_sold_date_sk = d.d_date_sk
WHERE d.d_year BETWEEN 1998 AND 2002
UNION ALL
SELECT ws_quantity * ws_list_price AS sales_value
FROM web_sales ws
JOIN date_ranges d ON ws.ws_sold_date_sk = d.d_date_sk
WHERE d.d_year BETWEEN 1998 AND 2002
) sales_data
),
last_year_sales AS (
SELECT
'store' AS channel,
i.i_brand_id,
i.i_class_id,
i.i_category_id,
SUM(ss.ss_quantity * ss.ss_list_price) AS sales,
COUNT(*) AS number_sales
FROM store_sales ss
JOIN common_items i ON ss.ss_item_sk = i.i_item_sk
JOIN date_ranges d ON ss.ss_sold_date_sk = d.d_date_sk
JOIN (SELECT MAX(last_year_week) AS week_seq FROM target_weeks) tw
ON d.d_week_seq = tw.week_seq
GROUP BY i.i_brand_id, i.i_class_id, i.i_category_id
),
this_year_sales AS (
SELECT
'store' AS channel,
i.i_brand_id,
i.i_class_id,
i.i_category_id,
SUM(ss.ss_quantity * ss.ss_list_price) AS sales,
COUNT(*) AS number_sales
FROM store_sales ss
JOIN common_items i ON ss.ss_item_sk = i.i_item_sk
JOIN date_ranges d ON ss.ss_sold_date_sk = d.d_date_sk
JOIN (SELECT MAX(this_year_week) AS week_seq FROM target_weeks) tw
ON d.d_week_seq = tw.week_seq
GROUP BY i.i_brand_id, i.i_class_id, i.i_category_id
)
SELECT
ty.channel AS ty_channel,
ty.i_brand_id AS ty_brand,
ty.i_class_id AS ty_class,
ty.i_category_id AS ty_category,
ty.sales AS ty_sales,
ty.number_sales AS ty_number_sales,
ly.channel AS ly_channel,
ly.i_brand_id AS ly_brand,
ly.i_class_id AS ly_class,
ly.i_category_id AS ly_category,
ly.sales AS ly_sales,
ly.number_sales AS ly_number_sales
FROM this_year_sales ty
JOIN last_year_sales ly ON ty.i_brand_id = ly.i_brand_id
AND ty.i_class_id = ly.i_class_id
AND ty.i_category_id = ly.i_category_id
JOIN avg_sales ON ty.sales > avg_sales.average_sales AND ly.sales > avg_sales.average_sales
ORDER BY
ty.channel,
ty.i_brand_id,
ty.i_class_id,
ty.i_category_id
LIMIT 100;
```

## 5. Recommendations

1. **Cache Small Tables**: Implement CACHE TABLE for item and date_dim tables
   ```sql
   CACHE TABLE tpcds.tpcds_sf1000_delta_lc.item;
   CACHE TABLE tpcds.tpcds_sf1000_delta_lc.date_dim;
   ```

2. **Maintain Current Clustering Keys**: The current clustering keys for large tables (store_sales, catalog_sales, web_sales) are already optimal

3. **Improve Cache Efficiency**: Consider increasing cache allocation to improve the current 53.7% cache efficiency

4. **Reduce Shuffle Operations**: Consider using broadcast hints for smaller tables in joins
   ```sql
   -- Example hint usage
   SELECT /*+ BROADCAST(small_table) */ * FROM large_table JOIN small_table ON ...
   ```

5. **Monitor Query Performance**: Continue monitoring execution time and data read volume after implementing these recommendations