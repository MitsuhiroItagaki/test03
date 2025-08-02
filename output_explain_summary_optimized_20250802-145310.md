# EXPLAIN + EXPLAIN COST要約結果 (optimized)

## 📊 基本情報
- 生成日時: 2025-08-02 14:53:10
- クエリタイプ: optimized
- 元サイズ: EXPLAIN(69,944文字) + EXPLAIN COST(493,435文字) = 563,379文字
- 要約後サイズ: 2,412文字
- 圧縮率: 233x

## 🧠 LLM要約結果

# Databricks SQLクエリパフォーマンス分析

## 📊 Physical Plan要約

### 主要な処理ステップ
1. **複数テーブルからのデータ取得**: store_sales, catalog_sales, web_sales, date_dim, itemテーブルから必要なデータを読み込み
2. **サブクエリ実行**: 平均売上を計算するサブクエリ（複数のテーブルからのUNION ALL）
3. **フィルタリング**: 平均売上を超える商品のフィルタリング
4. **集計処理**: ブランド、クラス、カテゴリごとの売上集計
5. **JOIN処理**: 複数のJOIN操作（特にBroadcastHashJoinが多用）
6. **ソート**: ブランドID、クラスID、カテゴリIDでのソート
7. **LIMIT**: 最終結果を100行に制限

### JOIN方式とデータ移動パターン
- **主要JOIN方式**: BroadcastHashJoin（BuildRight, BuildLeft）が多用
- **データ移動**: 
  - PhotonShuffleExchangeSource/Sink による効率的なデータ移動
  - SinglePartitionによる集約処理
  - hashpartitioningによるデータ分散（1024パーティション）

### Photon利用状況
- **高度なPhoton活用**: PhotonProject, PhotonBroadcastHashJoin, PhotonFilter, PhotonGroupingAgg, PhotonTopKなど多数のPhoton最適化演算子を使用
- **AdaptiveSparkPlan**: 実行時の最適化が有効

## 💰 統計情報サマリー

### テーブルサイズと行数
- **store_sales**: 407.7 GiB, 約28.8億行
- **最終結果セット**: 11.1 KiB, 100行（LIMIT適用後）
- **中間結果**: 約14.9 MiB, 約13.7万行（ソート前）

### JOIN選択率とフィルタ効率
- **date_dim フィルタ**: 年度条件（1998-2002）により、約1,461行に絞り込み（高効率）
- **サブクエリ結果**: 平均売上計算のサブクエリは単一行を返却
- **メインクエリフィルタ**: 平均売上を超える商品に絞り込み（約13.7万行に削減）

### カラム統計
- **ブランドID (i_brand_id)**: 858種類の異なる値（1001001〜10016017）
- **クラスID (i_class_id)**: 16種類の異なる値（1〜16）
- **カテゴリID (i_category_id)**: 10種類の異なる値（1〜10）
- **数量 (ss_quantity)**: 1〜100の範囲、99種類の異なる値

### パーティション分散状況
- **ハッシュパーティショニング**: ブランドID、クラスID、カテゴリIDに基づく1024パーティション
- **シングルパーティション**: 集約処理や最終結果の収集に使用

## ⚡ パフォーマンス分析

### 実行コストの内訳
1. **最もコストが高い操作**: store_sales（407.7 GiB）からのスキャン
2. **サブクエリコスト**: 複数テーブル（store_sales, catalog_sales, web_sales）からのUNION ALL処理
3. **JOIN処理**: 複数のBroadcastHashJoinによるコスト
4. **集計処理**: GroupingAggによる集計コスト

### ボトルネックになりそうな操作
1. **大規模テーブルスキャン**: store_sales（407.7 GiB）のスキャンが最大のボトルネック
2. **複数テーブルUNION**: サブクエリでの3つの販売テーブル（store_sales, catalog_sales, web_sales）の統合
3. **シャッフル操作**: hashpartitioningによるデータ再分散

### 最適化の余地がある箇所
1. **パーティションプルーニング**: date_dimテーブルのフィルタリングは効果的だが、さらに販売テーブルのパーティション最適化が可能
2. **JOIN順序**: 複数のJOIN操作の順序最適化
3. **フィルタプッシュダウン**: 動的フィルタリング（dynamicpruning）が使用されているが、さらに最適化の余地あり
4. **カラム選択**: 必要なカラムのみを早期に選択することでデータ移動量を削減可能
5. **メモリ使用量**: BroadcastHashJoinのビルド側のサイズ最適化

### 特記事項
- **Photon活用**: クエリ全体でPhoton最適化が効果的に適用されている
- **統計情報**: カラム統計が適切に収集されており、オプティマイザの判断に貢献
- **動的フィルタリング**: dynamicpruningが適用され、不要なデータ読み込みを回避
- **アダプティブ実行**: AdaptiveSparkPlanが有効で、実行時の最適化が期待できる

このクエリは複雑なJOINと集計を含むが、Photon最適化とブロードキャストJOINの効果的な使用により、比較的効率的に実行されると予測されます。最大のボトルネックは大規模テーブルのスキャンとデータ移動にあります。

## 💰 統計情報抽出

## 📊 統計情報サマリー（簡潔版）
- **総統計項目数**: 210個
- **テーブル統計**: 208個
- **パーティション情報**: 2個

### 🎯 主要統計
📊 テーブルサイズ: :  :     +- Union false, false, Statistics(sizeInBytes=61.7 GiB, rowCount=4.14E+9, ColumnStat: N/A)...
📊 テーブルサイズ: :  :        :- Project [(cast(ss_quantity#85127 as decimal(10,0)) * ss_list_price#85129) AS sales_am...
📊 テーブルサイズ: :  :        :  +- Join Inner, (ss_sold_date_sk#85117 = d_date_sk#85140), rightHint=(dynamicPruningFi...
📊 テーブルサイズ: :  :        :     :- Project [ss_sold_date_sk#85117, ss_quantity#85127, ss_list_price#85129], Statis...
📊 テーブルサイズ: :  :        :     :  +- Filter (isnotnull(ss_sold_date_sk#85117) AND dynamicpruning#85737 85735), St...

💡 詳細な統計情報は DEBUG_ENABLED='Y' で確認できます
