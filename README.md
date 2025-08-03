# Databricks SQLプロファイラー分析ツール

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Compatible-red)](https://databricks.com/)

Databricks SQLクエリのパフォーマンス分析とAI駆動型最適化提案を提供する包括的なツールです。

## 🚀 概要

このツールは`databricks_sql_profiler_analysis_en.py`（14,220行）で構成される単一ファイルのDatabricksノートブックで、SQLプロファイラーJSONログを分析し、詳細なパフォーマンス洞察とLLMによる最適化提案を提供します。

## ✨ 主要機能

- **📊 包括的パフォーマンス分析**: Databricks実行計画メトリクスの深度分析
- **🤖 AI駆動型最適化**: 複数LLM対応（Databricks、OpenAI、Azure、Anthropic）
- **🔍 Liquid Clustering分析**: 高度なクラスタリングキー推奨
- **📋 多言語レポート**: 英語・日本語対応の詳細分析レポート

## 📋 ファイル構成

```
databricks_sql_profiler_analysis_en.py (14,220行)
├── 設定・セットアップ (1-362行)
├── JSONファイル読み込み (363-424行)
├── パフォーマンスメトリクス抽出 (425-1751行)
├── Liquid Clustering分析 (1752-3364行)
├── LLM統合・分析 (3365-5228行)
├── クエリ最適化エンジン (5229-10990行)
├── レポート生成 (10991-13717行)
└── ファイル管理・推敲 (13718-14220行)
```

## 🚀 クイックスタート

1. **データ準備**: DatabricksでSQLクエリを実行し、プロファイラーJSONファイルをダウンロード
2. **設定**: `JSON_FILE_PATH`とLLM設定を更新
3. **実行**: Databricksノートブックのセルを順次実行
4. **結果確認**: 生成されたレポートと最適化SQLを確認

## 📊 サンプル出力

```
実行時間: 29.8秒 (良好)
データ読み取り量: 34.85GB (大容量)
Photon利用: 有効 ✅
キャッシュ効率: 15.4% (要改善)
フィルタ効率: 90.1% (良好)
```

## 📁 詳細ドキュメント

- **[README_EN.md](README_EN.md)**: English documentation
- **[README_JP.md](README_JP.md)**: 日本語詳細ドキュメント

## 📄 ライセンス

MIT License

## 👨‍💻 作成者

- MitsuhiroItagaki

---

**Databricksコミュニティのために❤️で作成**