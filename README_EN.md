# Databricks SQL Profiler Analysis Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Compatible-red)](https://databricks.com/)

A comprehensive AI-powered tool for analyzing Databricks SQL query performance and providing automated optimization recommendations.

## 🚀 Overview

This tool analyzes Databricks SQL profiler JSON log files and provides detailed performance insights, bottleneck identification, and specific optimization recommendations using Large Language Models (LLMs). It's designed to help data engineers and analysts optimize their SQL queries for better performance on Databricks platforms.

## ✨ Key Features

### 📊 **Comprehensive Performance Analysis**
- **SQL Profiler JSON Analysis**: Deep analysis of Databricks execution plan metrics
- **Bottleneck Identification**: Automatic detection of performance bottlenecks
- **Multi-dimensional Metrics**: Execution time, data volume, cache efficiency, shuffle operations
- **Filter Rate Analysis**: Detailed analysis of data pruning efficiency

### 🤖 **AI-Powered Optimization**
- **Multi-LLM Support**: Databricks, OpenAI, Azure OpenAI, Anthropic Claude
- **Intelligent Recommendations**: Context-aware optimization suggestions
- **Liquid Clustering Analysis**: Advanced clustering key recommendations
- **Query Optimization**: Automated SQL query improvement suggestions

### 📋 **Comprehensive Reporting**
- **Detailed Analysis Reports**: Multi-language support (English/Japanese)
- **Performance Metrics**: Visual performance indicators and comparisons
- **Implementation Guides**: Step-by-step optimization implementation
- **SQL Code Generation**: Ready-to-use optimized SQL queries

### 🔧 **Advanced Features**
- **EXPLAIN Plan Analysis**: Automated execution plan comparison
- **Broadcast Join Optimization**: Smart broadcast hint recommendations
- **Performance Comparison**: Before/after optimization analysis
- **Debug Mode**: Detailed intermediate file preservation

## 🏗️ Architecture

```
databricks_sql_profiler_analysis_en.py (14,220 lines)
├── Configuration & Setup
├── JSON Profiler Loading
├── Performance Metrics Extraction
├── Bottleneck Analysis
├── LLM-Based Optimization
├── Liquid Clustering Analysis
├── Query Optimization Engine
├── Report Generation
└── File Management
```

## 📋 Prerequisites

### Required Environment
- **Databricks Runtime**: 10.4 LTS or higher
- **Python**: 3.8+
- **Libraries**: `requests`, `pyspark`, `json`, `re`

### LLM Configuration (Choose One)
- **Databricks Model Serving**: Claude 3.7 Sonnet endpoint
- **OpenAI API**: GPT-4o or GPT-4 Turbo
- **Azure OpenAI**: GPT-4 deployment
- **Anthropic API**: Claude 3.5 Sonnet

## ⚙️ Configuration

### 1. File Path Configuration
```python
# Set your SQL profiler JSON file path
JSON_FILE_PATH = 'your-profile-file.json'

# Language setting
OUTPUT_LANGUAGE = 'en'  # 'en' for English, 'ja' for Japanese

# Enable EXPLAIN statement execution
EXPLAIN_ENABLED = 'Y'  # 'Y' to enable, 'N' to disable

# Debug mode
DEBUG_ENABLED = 'Y'  # 'Y' to preserve intermediate files
```

### 2. LLM Endpoint Configuration
```python
LLM_CONFIG = {
    "provider": "databricks",  # databricks, openai, azure_openai, anthropic
    
    "databricks": {
        "endpoint_name": "databricks-claude-3-7-sonnet",
        "max_tokens": 131072,
        "temperature": 0.0
    },
    
    "openai": {
        "api_key": "your-openai-api-key",
        "model": "gpt-4o",
        "max_tokens": 16000,
        "temperature": 0.0
    }
    # ... other providers
}
```

### 3. Database Configuration
```python
CATALOG = 'your_catalog'
DATABASE = 'your_database'
```

## 🚀 Quick Start

### Step 1: Prepare Your Data
1. Execute your SQL query in Databricks
2. Download the query profiler JSON file
3. Place the file in your workspace

### Step 2: Configure the Tool
```python
# Update configuration
JSON_FILE_PATH = 'your-profile-file.json'
OUTPUT_LANGUAGE = 'en'
LLM_CONFIG["provider"] = "databricks"  # or your preferred provider
```

### Step 3: Run Analysis
Execute all cells in the Databricks notebook sequentially:
1. **Configuration cells** (1-5): Set up parameters
2. **Analysis cells** (6-42): Extract metrics and analyze
3. **Optimization cells** (43-45): Generate recommendations
4. **Report cells** (46): Create final report

### Step 4: Review Results
Check generated files:
- `output_final_report_en_TIMESTAMP.md`: Comprehensive analysis report
- `output_optimized_query_TIMESTAMP.sql`: Optimized SQL query
- `output_original_query_TIMESTAMP.sql`: Original query for comparison

## 📊 Sample Output

### Performance Metrics
```
Execution Time: 29.8 seconds (Good)
Data Read Volume: 34.85GB (Large)
Photon Utilization: Enabled ✅
Cache Efficiency: 15.4% (Needs Improvement)
Filter Efficiency: 90.1% (Good)
```

### Optimization Recommendations
- **Liquid Clustering**: Configure clustering keys for better data pruning
- **Broadcast Joins**: Optimize small table broadcast strategies
- **Cache Optimization**: Improve cache hit rates for frequently accessed data
- **Query Rewriting**: Structural improvements for better performance

## 🔧 Advanced Usage

### Multiple LLM Providers
```python
# Switch to OpenAI
LLM_CONFIG["provider"] = "openai"

# Switch to Anthropic
LLM_CONFIG["provider"] = "anthropic"

# Switch to Azure OpenAI
LLM_CONFIG["provider"] = "azure_openai"
```

### Custom Analysis Parameters
```python
# Enable extended thinking mode (Databricks Claude only)
LLM_CONFIG["databricks"]["thinking_enabled"] = True
LLM_CONFIG["databricks"]["thinking_budget_tokens"] = 65536
```

### Batch Processing
```python
# Analyze multiple profile files
profile_files = ['profile1.json', 'profile2.json', 'profile3.json']
for file_path in profile_files:
    JSON_FILE_PATH = file_path
    # Run analysis...
```

## 📁 Output Files

| File Type | Description | Example |
|-----------|-------------|---------|
| **Final Report** | Comprehensive analysis report | `output_final_report_en_20250803-145134.md` |
| **Optimized Query** | AI-optimized SQL query | `output_optimized_query_20250803-144903.sql` |
| **Original Query** | Original query for comparison | `output_original_query_20250803-144903.sql` |
| **EXPLAIN Results** | Execution plan analysis | `output_explain_optimized_*.txt` |
| **Debug Files** | Intermediate analysis files | Various `_debug.json` files |

## 🎯 Use Cases

### 1. **Query Performance Troubleshooting**
- Identify slow-running queries
- Pinpoint performance bottlenecks
- Get specific optimization recommendations

### 2. **Liquid Clustering Optimization**
- Analyze current clustering effectiveness
- Get clustering key recommendations
- Generate implementation SQL

### 3. **Cost Optimization**
- Reduce data scanning costs
- Optimize compute resource usage
- Improve overall query efficiency

### 4. **Performance Monitoring**
- Regular performance health checks
- Benchmark query improvements
- Track optimization progress

## 🛠️ Troubleshooting

### Common Issues

**1. File Not Found**
```
❌ Error: Profile file not found
Solution: Check file path and ensure JSON file exists
```

**2. LLM Configuration Error**
```
❌ Error: LLM provider not configured
Solution: Set valid API keys and endpoint configurations
```

**3. Memory Issues**
```
❌ Error: Analysis failed due to large file size
Solution: Enable DEBUG_ENABLED='N' to reduce memory usage
```

### Debug Mode
Enable debug mode to preserve intermediate files:
```python
DEBUG_ENABLED = 'Y'
```

This preserves:
- Raw analysis data
- Intermediate calculations
- LLM response details
- Error logs

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch
3. **Test** your changes thoroughly
4. **Submit** a pull request with clear descriptions

### Development Setup
```bash
git clone https://github.com/MitsuhiroItagaki/test03.git
cd test03
# Set up your Databricks environment
# Configure LLM endpoints
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Databricks Team**: For the excellent SQL profiling capabilities
- **LLM Providers**: OpenAI, Anthropic, Azure for powerful AI capabilities
- **Community Contributors**: For feedback and improvements

## 📞 Support

For issues and questions:
- **GitHub Issues**: [Create an issue](https://github.com/MitsuhiroItagaki/test03/issues)
- **Documentation**: Check this README and code comments
- **Examples**: Review sample output files in the repository

---

**Made with ❤️ for the Databricks community**