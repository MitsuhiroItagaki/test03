# Databricks SQL Performance Optimization Report

## 1. Executive Summary

Query execution time is 29.8 seconds, which is good, but several optimization opportunities were identified:

| Metric | Current Status | Evaluation | Priority |
|--------|---------------|------------|----------|
| Execution Time | 29.8s | ✅ Good | - |
| Data Read Volume | 34.85GB | ⚠️ Large Volume | High |
| Photon Enabled | Yes | ✅ Good | - |
| Shuffle Operations | 64 times | ⚠️ High | Medium |
| Spill Occurrence | None | ✅ Good | - |
| Cache Efficiency | 15.4% | ⚠️ Low | Medium |
| Filter Efficiency | 90.1% | ✅ Good | - |
| Data Skew | AQE Handled | ✅ Good | - |

## 2. Time-Consuming Operations Analysis

### Top 10 Time-Consuming Processes

1. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.catalog_sales)**
   - Execution Time: 110.5s (11.9% of total)
   - Peak Memory: 2455.7 MB
   - Filter rate: 0.0% (read: 121.20GB, pruned: 0.00GB)
   - Current clustering key: cs_item_sk, cs_sold_date_sk

2. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.store_sales)**
   - Execution Time: 90.0s (9.7% of total)
   - Peak Memory: 2731.7 MB
   - Filter rate: 1.4% (read: 159.60GB, pruned: 2.25GB)
   - Current clustering key: ss_sold_date_sk, ss_item_sk

3. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.store_sales)**
   - Execution Time: 78.4s (8.4% of total)
   - Peak Memory: 2725.1 MB
   - Filter rate: 1.4% (read: 159.60GB, pruned: 2.25GB)
   - Current clustering key: ss_sold_date_sk, ss_item_sk

4. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.web_sales)**
   - Execution Time: 74.3s (8.0% of total)
   - Peak Memory: 3415.1 MB
   - Filter rate: 0.0% (read: 60.17GB, pruned: 0.00GB)
   - Current clustering key: ws_item_sk, ws_sold_date_sk

5. **Photon Grouping Aggregate**
   - Execution Time: 58.3s (6.3% of total)
   - Peak Memory: 715.5 MB

6. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.catalog_sales)**
   - Execution Time: 53.2s (5.7% of total)
   - Peak Memory: 2436.4 MB
   - Filter rate: 0.0% (read: 121.20GB, pruned: 0.00GB)
   - Current clustering key: cs_item_sk, cs_sold_date_sk

7. **Photon Inner Join**
   - Execution Time: 47.6s (5.1% of total)
   - Peak Memory: 198.0 MB

8. **Photon Data Source Scan (tpcds.tpcds_sf1000_delta_lc.web_sales)**
   - Execution Time: 45.0s (4.8% of total)
   - Peak Memory: 3500.4 MB
   - Filter rate: 0.0% (read: 60.17GB, pruned: 0.00GB)
   - Current clustering key: ws_item_sk, ws_sold_date_sk

9. **Photon Grouping Aggregate**
   - Execution Time: 37.7s (4.1% of total)
   - Peak Memory: 641.8 MB

10. **Photon Aggregate**
    - Execution Time: 32.1s (3.5% of total)
    - Peak Memory: 0.0 MB

## 3. Liquid Clustering Recommendations

### High-Priority Tables

#### 1. store_sales Table
- **Table Size**: 159.60GB
- **Current clustering key**: Not configured
- **Recommended clustering columns**: ss_sold_date_sk, ss_item_sk
- **Filter rate**: 90.10% (read: 159.60GB, pruned: 143.64GB)

```sql
ALTER TABLE tpcds.tpcds_sf1000_delta_lc.store_sales 
CLUSTER BY (ss_sold_date_sk, ss_item_sk);
OPTIMIZE tpcds.tpcds_sf1000_delta_lc.store_sales FULL;
```

**Rationale**:
- Large table size (159.60GB) makes clustering highly beneficial
- ss_sold_date_sk and ss_item_sk are used in JOIN and filter conditions
- Note: In Liquid Clustering, column order doesn't affect node-level data locality
- Expected benefit: Improved scan efficiency and file pruning

#### 2. catalog_sales Table
- **Table Size**: 121.20GB
- **Current clustering key**: Not configured
- **Recommended clustering columns**: cs_sold_date_sk, cs_item_sk
- **Filter rate**: Information not available (estimated: 85-90%)

```sql
ALTER TABLE tpcds.tpcds_sf1000_delta_lc.catalog_sales 
CLUSTER BY (cs_sold_date_sk, cs_item_sk);
OPTIMIZE tpcds.tpcds_sf1000_delta_lc.catalog_sales FULL;
```

**Rationale**:
- Large table size (121.20GB) makes clustering highly beneficial
- Similar access patterns to store_sales table
- Expected benefit: Improved scan efficiency and file pruning

#### 3. web_sales Table
- **Table Size**: 60.17GB
- **Current clustering key**: Not configured
- **Recommended clustering columns**: ws_sold_date_sk, ws_item_sk
- **Filter rate**: Information not available (estimated: 85-90%)

```sql
ALTER TABLE tpcds.tpcds_sf1000_delta_lc.web_sales 
CLUSTER BY (ws_sold_date_sk, ws_item_sk);
OPTIMIZE tpcds.tpcds_sf1000_delta_lc.web_sales FULL;
```

**Rationale**:
- Significant table size (60.17GB) makes clustering beneficial
- Similar access patterns to other sales tables
- Expected benefit: Improved scan efficiency and file pruning

### Not Recommended for Clustering

#### 4. item Table
- **Table Size**: 0.03GB
- **Current clustering key**: Not configured
- **Recommendation**: Clustering not recommended due to small size

```sql
-- Liquid Clustering not recommended due to limited benefit
-- Alternative: CREATE INDEX ON tpcds.tpcds_sf1000_delta_lc.item (i_item_sk);
```

**Rationale**:
- Very small table size (0.03GB) limits clustering benefits
- Consider using indexes or memory caching instead

#### 5. date_dim Table
- **Table Size**: 0.00GB
- **Current clustering key**: Not configured
- **Recommendation**: Clustering not recommended due to minimal size

```sql
-- Liquid Clustering not recommended due to limited benefit
-- Alternative: CREATE INDEX ON tpcds.tpcds_sf1000_delta_lc.date_dim (d_date_sk, d_year, d_moy);
```

**Rationale**:
- Extremely small table size makes clustering ineffective
- Consider using indexes or memory caching instead

## 4. Query Optimization Analysis

### Optimization Trials Summary
- **Number of trials**: 3 attempts
- **Selected query**: Original query (no improvements achieved)
- **Reason**: Optimization attempts did not yield performance improvements

### Original Query (Recommended)

```sql
USE CATALOG tpcds;
USE SCHEMA tpcds_sf1000_delta_lc;
with cross_items as (
select
i_item_sk ss_item_sk
from
item,
(
select
iss.i_brand_id brand_id,
iss.i_class_id class_id,
iss.i_category_id category_id
from
store_sales,
item iss,
date_dim d1
where
ss_item_sk = iss.i_item_sk
and ss_sold_date_sk = d1.d_date_sk
and d1.d_year between 1998
AND 1998 + 4
intersect
select
ics.i_brand_id,
ics.i_class_id,
ics.i_category_id
from
catalog_sales,
item ics,
date_dim d2
where
cs_item_sk = ics.i_item_sk
and cs_sold_date_sk = d2.d_date_sk
and d2.d_year between 1998
AND 1998 + 4
intersect
select
iws.i_brand_id,
iws.i_class_id,
iws.i_category_id
from
web_sales,
item iws,
date_dim d3
where
ws_item_sk = iws.i_item_sk
and ws_sold_date_sk = d3.d_date_sk
and d3.d_year between 1998
AND 1998 + 4
)
where
i_brand_id = brand_id
and i_class_id = class_id
and i_category_id = category_id
),
avg_sales as (
select
avg(quantity * list_price) average_sales
from
(
select
ss_quantity quantity,
ss_list_price list_price
from
store_sales,
date_dim
where
ss_sold_date_sk = d_date_sk
and d_year between 1998
and 1998 + 4
union all
select
cs_quantity quantity,
cs_list_price list_price
from
catalog_sales,
date_dim
where
cs_sold_date_sk = d_date_sk
and d_year between 1998
and 1998 + 4
union all
select
ws_quantity quantity,
ws_list_price list_price
from
web_sales,
date_dim
where
ws_sold_date_sk = d_date_sk
and d_year between 1998
and 1998 + 4
) x
)
```

**Note**: The complete optimized query is available in file: `output_optimized_query_20250803-134841.sql`

## 5. Key Recommendations

1. **Implement Liquid Clustering** for the three large sales tables:
   - store_sales (159.60GB)
   - catalog_sales (121.20GB)
   - web_sales (60.17GB)

2. **Use indexes instead of clustering** for small dimension tables:
   - item (0.03GB)
   - date_dim (0.00GB)

3. **Maintain the original query** as optimization attempts did not yield improvements

4. **Consider cache warming** for frequently accessed small tables to improve the low cache efficiency (15.4%)