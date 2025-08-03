# Databricks SQLプロファイラー分析ツール

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Compatible-red)](https://databricks.com/)

Databricks SQLクエリのパフォーマンス分析とAI駆動型最適化提案を提供する包括的なツールです。

## 🚀 概要

このツールは`databricks_sql_profiler_analysis_en.py`という単一のPythonファイル（14,220行）で構成されており、Databricks SQLプロファイラーJSONログファイルを分析し、詳細なパフォーマンス洞察、ボトルネック特定、および大規模言語モデル（LLM）を使用した具体的な最適化提案を提供します。

## 📋 `databricks_sql_profiler_analysis_en.py` ファイル詳細

### 🏗️ ファイル構造（14,220行）

```
databricks_sql_profiler_analysis_en.py
├── 🔧 設定・セットアップセクション (1-362行)
│   ├── ファイルパス設定
│   ├── LLMエンドポイント設定
│   └── 基本環境設定
├── 📂 JSONファイル読み込み機能 (363-424行)
├── 📊 パフォーマンスメトリクス抽出 (425-1751行)
│   ├── 基本メトリクス抽出
│   ├── ボトルネック指標計算
│   └── 詳細分析機能
├── 🔍 Liquid Clustering分析 (1752-3364行)
│   ├── クラスタリングデータ抽出
│   ├── 最適化機会分析
│   └── SQL実装生成
├── 🤖 LLM統合・分析エンジン (3365-5228行)
│   ├── 複数LLMプロバイダー対応
│   ├── ボトルネック分析
│   └── 最適化提案生成
├── 🔄 クエリ最適化エンジン (5229-10990行)
│   ├── クエリ抽出・分析
│   ├── 最適化クエリ生成
│   └── パフォーマンス比較
├── 📝 レポート生成システム (10991-13717行)
│   ├── 包括的レポート生成
│   ├── 多言語対応
│   └── SQLコード整形
└── 🧹 ファイル管理・最終処理 (13718-14220行)
    ├── レポート推敲
    ├── ファイル整理
    └── デバッグモード処理
```

### ✨ 主要機能

#### 📊 **包括的パフォーマンス分析**
- **SQLプロファイラーJSON分析**: Databricks実行計画メトリクスの深度分析
- **ボトルネック特定**: パフォーマンスボトルネックの自動検出
- **多次元メトリクス**: 実行時間、データ量、キャッシュ効率、シャッフル操作
- **フィルタ率分析**: データプルーニング効率の詳細分析

#### 🤖 **AI駆動型最適化**
- **マルチLLM対応**: Databricks、OpenAI、Azure OpenAI、Anthropic Claude
- **インテリジェント推奨**: コンテキスト認識型最適化提案
- **Liquid Clustering分析**: 高度なクラスタリングキー推奨
- **クエリ最適化**: 自動SQLクエリ改善提案

#### 📋 **包括的レポート機能**
- **詳細分析レポート**: 多言語サポート（英語/日本語）
- **パフォーマンス指標**: 視覚的パフォーマンス指標と比較
- **実装ガイド**: ステップバイステップの最適化実装
- **SQLコード生成**: すぐに使用可能な最適化SQLクエリ

## ⚙️ 設定方法

### 1. ファイルパス設定
```python
# SQLプロファイラーJSONファイルのパスを設定
JSON_FILE_PATH = 'query-profile_01f0703c-c975-1f48-ad71-ba572cc57272.json'

# 言語設定
OUTPUT_LANGUAGE = 'en'  # 'en'は英語、'ja'は日本語

# EXPLAIN文実行設定
EXPLAIN_ENABLED = 'Y'  # 'Y'で有効、'N'で無効

# デバッグモード
DEBUG_ENABLED = 'Y'  # 'Y'で中間ファイル保持
```

### 2. LLMエンドポイント設定
```python
LLM_CONFIG = {
    "provider": "databricks",  # databricks, openai, azure_openai, anthropic
    
    "databricks": {
        "endpoint_name": "databricks-claude-3-7-sonnet",
        "max_tokens": 131072,  # 128Kトークン（Claude 3.7 Sonnet最大制限）
        "temperature": 0.0,
        "thinking_enabled": False  # 拡張思考モード（高速実行優先）
    },
    
    "openai": {
        "api_key": "your-openai-api-key",
        "model": "gpt-4o",
        "max_tokens": 16000,
        "temperature": 0.0
    }
    # ... 他のプロバイダー
}
```

### 3. データベース設定
```python
CATALOG = 'tpcds'
DATABASE = 'tpcds_sf1000_delta_lc'
```

## 🚀 使用方法

### ステップ1: データ準備
1. DatabricksでSQLクエリを実行
2. クエリプロファイラーJSONファイルをダウンロード
3. ワークスペースにファイルを配置

### ステップ2: ツール設定
```python
# 設定を更新
JSON_FILE_PATH = 'your-profile-file.json'
OUTPUT_LANGUAGE = 'ja'  # 日本語出力の場合
LLM_CONFIG["provider"] = "databricks"  # または希望のプロバイダー
```

### ステップ3: 分析実行
Databricksノートブックのセルを順次実行：
1. **設定セル** (1-5): パラメータ設定
2. **分析セル** (6-42): メトリクス抽出と分析
3. **最適化セル** (43-45): 推奨事項生成
4. **レポートセル** (46): 最終レポート作成

### ステップ4: 結果確認
生成されたファイルを確認：
- `output_final_report_jp_TIMESTAMP.md`: 包括的分析レポート
- `output_optimized_query_TIMESTAMP.sql`: 最適化SQLクエリ
- `output_original_query_TIMESTAMP.sql`: 比較用元クエリ

## 📊 サンプル出力

### パフォーマンス指標
```
実行時間: 29.8秒 (良好)
データ読み取り量: 34.85GB (大容量)
Photon利用: 有効 ✅
キャッシュ効率: 15.4% (要改善)
フィルタ効率: 90.1% (良好)
```

### 最適化推奨事項
- **Liquid Clustering**: より良いデータプルーニングのためのクラスタリングキー設定
- **ブロードキャストジョイン**: 小テーブルのブロードキャスト戦略最適化
- **キャッシュ最適化**: 頻繁アクセスデータのキャッシュヒット率向上
- **クエリ書き換え**: より良いパフォーマンスのための構造的改善

## 🔧 主要関数の説明

### パフォーマンス分析関数
```python
def extract_performance_metrics(profiler_data: Dict[str, Any]) -> Dict[str, Any]:
    """プロファイラーデータからパフォーマンスメトリクスを抽出"""

def calculate_bottleneck_indicators(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """ボトルネック指標を計算"""

def extract_detailed_bottleneck_analysis(extracted_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """詳細なボトルネック分析を実行"""
```

### Liquid Clustering分析関数
```python
def extract_liquid_clustering_data(profiler_data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Liquid Clusteringデータを抽出"""

def analyze_liquid_clustering_opportunities(profiler_data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Liquid Clustering最適化機会を分析"""
```

### LLM統合関数
```python
def analyze_bottlenecks_with_llm(metrics: Dict[str, Any]) -> str:
    """LLMを使用してボトルネックを分析"""

def generate_optimized_query_with_llm(original_query: str, analysis_result: str, metrics: Dict[str, Any]) -> str:
    """LLMを使用して最適化クエリを生成"""
```

### レポート生成関数
```python
def generate_comprehensive_optimization_report(query_id: str, optimized_result: str, metrics: Dict[str, Any], analysis_result: str = "", performance_comparison: Dict[str, Any] = None, best_attempt_num: int = 1) -> str:
    """包括的最適化レポートを生成"""

def refine_report_content_with_llm(report_content: str) -> str:
    """LLMを使用してレポートを推敲"""
```

## 📁 出力ファイル

| ファイル種類 | 説明 | 例 |
|-------------|------|-----|
| **最終レポート** | 包括的分析レポート | `output_final_report_jp_20250803-145134.md` |
| **最適化クエリ** | AI最適化SQLクエリ | `output_optimized_query_20250803-144903.sql` |
| **元クエリ** | 比較用元クエリ | `output_original_query_20250803-144903.sql` |
| **EXPLAIN結果** | 実行計画分析 | `output_explain_optimized_*.txt` |
| **デバッグファイル** | 中間分析ファイル | 各種 `_debug.json` ファイル |

## 🎯 使用ケース

### 1. **クエリパフォーマンストラブルシューティング**
- 低速クエリの特定
- パフォーマンスボトルネックの特定
- 具体的な最適化推奨事項の取得

### 2. **Liquid Clustering最適化**
- 現在のクラスタリング効率の分析
- クラスタリングキー推奨事項の取得
- 実装SQLの生成

### 3. **コスト最適化**
- データスキャンコストの削減
- コンピュートリソース使用量の最適化
- 全体的なクエリ効率の向上

### 4. **パフォーマンス監視**
- 定期的なパフォーマンスヘルスチェック
- クエリ改善のベンチマーク
- 最適化進捗の追跡

## 🛠️ トラブルシューティング

### よくある問題

**1. ファイルが見つからない**
```
❌ エラー: プロファイルファイルが見つかりません
解決策: ファイルパスを確認し、JSONファイルが存在することを確認
```

**2. LLM設定エラー**
```
❌ エラー: LLMプロバイダーが設定されていません
解決策: 有効なAPIキーとエンドポイント設定を設定
```

**3. メモリ問題**
```
❌ エラー: 大きなファイルサイズによる分析失敗
解決策: DEBUG_ENABLED='N'を有効にしてメモリ使用量を削減
```

### デバッグモード
中間ファイルを保持するためにデバッグモードを有効化：
```python
DEBUG_ENABLED = 'Y'
```

これにより以下が保持されます：
- 生分析データ
- 中間計算結果
- LLMレスポンス詳細
- エラーログ

## 🔧 高度な使用方法

### 複数LLMプロバイダーの切り替え
```python
# OpenAIに切り替え
LLM_CONFIG["provider"] = "openai"

# Anthropicに切り替え
LLM_CONFIG["provider"] = "anthropic"

# Azure OpenAIに切り替え
LLM_CONFIG["provider"] = "azure_openai"
```

### カスタム分析パラメータ
```python
# 拡張思考モードを有効化（Databricks Claude専用）
LLM_CONFIG["databricks"]["thinking_enabled"] = True
LLM_CONFIG["databricks"]["thinking_budget_tokens"] = 65536
```

### バッチ処理
```python
# 複数のプロファイルファイルを分析
profile_files = ['profile1.json', 'profile2.json', 'profile3.json']
for file_path in profile_files:
    JSON_FILE_PATH = file_path
    # 分析を実行...
```

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- **Databricksチーム**: 優れたSQLプロファイリング機能のために
- **LLMプロバイダー**: OpenAI、Anthropic、Azureの強力なAI機能のために
- **コミュニティ貢献者**: フィードバックと改善のために

## 📞 サポート

問題や質問について：
- **GitHub Issues**: [イシューを作成](https://github.com/MitsuhiroItagaki/test03/issues)
- **ドキュメント**: このREADMEとコードコメントを確認
- **例**: リポジトリ内のサンプル出力ファイルを確認

---

**Databricksコミュニティのために❤️で作成**