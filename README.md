# Databricks SQL Profiler Analysis Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Compatible-red)](https://databricks.com/)

A comprehensive tool for Databricks SQL query performance analysis and AI-driven optimization recommendations.

## 🚀 Overview

This tool consists of a single Python file `databricks_sql_profiler_analysis_en.py` (14,220 lines) that analyzes Databricks SQL profiler JSON log files and provides detailed performance insights, bottleneck identification, and specific optimization recommendations using Large Language Models (LLMs).

## 📋 `databricks_sql_profiler_analysis_en.py` File Details

### 🏗️ File Structure (14,220 lines)

```
databricks_sql_profiler_analysis_en.py
├── 🔧 Configuration & Setup Section (1-362 lines)
│   ├── File path configuration
│   ├── LLM endpoint configuration
│   └── Basic environment setup
├── 📂 JSON File Loading Functions (363-424 lines)
├── 📊 Performance Metrics Extraction (425-1751 lines)
│   ├── Basic metrics extraction
│   ├── Bottleneck indicator calculation
│   └── Detailed analysis functions
├── 🔍 Liquid Clustering Analysis (1752-3364 lines)
│   ├── Clustering data extraction
│   ├── Optimization opportunity analysis
│   └── SQL implementation generation
├── 🤖 LLM Integration & Analysis Engine (3365-5228 lines)
│   ├── Multi-LLM provider support
│   ├── Bottleneck analysis
│   └── Optimization recommendation generation
├── 🔄 Query Optimization Engine (5229-10990 lines)
│   ├── Query extraction & analysis
│   ├── Optimized query generation
│   └── Performance comparison
├── 📝 Report Generation System (10991-13717 lines)
│   ├── Comprehensive report generation
│   ├── Multi-language support
│   └── SQL code formatting
└── 🧹 File Management & Final Processing (13718-14220 lines)
    ├── Report refinement
    ├── File organization
    └── Debug mode handling
```

### ✨ Key Features

#### 📊 **Comprehensive Performance Analysis**
- **SQL Profiler JSON Analysis**: Deep analysis of Databricks execution plan metrics
- **Bottleneck Identification**: Automatic detection of performance bottlenecks
- **Multi-dimensional Metrics**: Execution time, data volume, cache efficiency, shuffle operations
- **Filter Rate Analysis**: Detailed analysis of data pruning efficiency

#### 🤖 **AI-Driven Optimization**
- **Multi-LLM Support**: Databricks, OpenAI, Azure OpenAI, Anthropic Claude
- **Intelligent Recommendations**: Context-aware optimization suggestions
- **Liquid Clustering Analysis**: Advanced clustering key recommendations
- **Query Optimization**: Automatic SQL query improvement suggestions

#### 📋 **Comprehensive Reporting**
- **Detailed Analysis Reports**: Multi-language support (English/Japanese)
- **Performance Metrics**: Visual performance indicators and comparisons
- **Implementation Guides**: Step-by-step optimization implementation
- **SQL Code Generation**: Ready-to-use optimized SQL queries

## ⚙️ Configuration Setup

### 1. File Path Configuration
```python
# Set the SQL profiler JSON file path
JSON_FILE_PATH = 'query-profile_01f0703c-c975-1f48-ad71-ba572cc57272.json'

# Language setting
OUTPUT_LANGUAGE = 'en'  # 'en' for English, 'ja' for Japanese

# EXPLAIN execution setting
EXPLAIN_ENABLED = 'Y'  # 'Y' to enable, 'N' to disable

# Debug mode
DEBUG_ENABLED = 'Y'  # 'Y' to retain intermediate files
```

### 2. LLM Endpoint Configuration
```python
LLM_CONFIG = {
    "provider": "databricks",  # databricks, openai, azure_openai, anthropic
    
    "databricks": {
        "endpoint_name": "databricks-claude-3-7-sonnet",
        "max_tokens": 131072,  # 128K tokens (Claude 3.7 Sonnet max limit)
        "temperature": 0.0,
        "thinking_enabled": False  # Extended thinking mode (fast execution priority)
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
CATALOG = 'tpcds'
DATABASE = 'tpcds_sf1000_delta_lc'
```

## 🚀 Usage Instructions

### Step 1: Data Preparation
1. Execute SQL query in Databricks
2. Download query profiler JSON file
3. Place file in workspace

### Step 2: Tool Configuration
```python
# Update configuration
JSON_FILE_PATH = 'your-profile-file.json'
OUTPUT_LANGUAGE = 'en'  # For English output
LLM_CONFIG["provider"] = "databricks"  # Or your preferred provider
```

### Step 3: Execute Analysis
Run Databricks notebook cells sequentially:
1. **Configuration Cells** (1-5): Parameter setup
2. **Analysis Cells** (6-42): Metrics extraction and analysis
3. **Optimization Cells** (43-45): Recommendation generation
4. **Report Cells** (46): Final report creation

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
- **Liquid Clustering**: Clustering key configuration for better data pruning
- **Broadcast Joins**: Small table broadcast strategy optimization
- **Cache Optimization**: Improve cache hit rates for frequently accessed data
- **Query Rewriting**: Structural improvements for better performance

## 🔧 Key Function Descriptions

### Performance Analysis Functions
```python
def extract_performance_metrics(profiler_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract performance metrics from profiler data"""

def calculate_bottleneck_indicators(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate bottleneck indicators"""

def extract_detailed_bottleneck_analysis(extracted_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Execute detailed bottleneck analysis"""
```

### Liquid Clustering Analysis Functions
```python
def extract_liquid_clustering_data(profiler_data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Liquid Clustering data"""

def analyze_liquid_clustering_opportunities(profiler_data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Liquid Clustering optimization opportunities"""
```

### LLM Integration Functions
```python
def analyze_bottlenecks_with_llm(metrics: Dict[str, Any]) -> str:
    """Analyze bottlenecks using LLM"""

def generate_optimized_query_with_llm(original_query: str, analysis_result: str, metrics: Dict[str, Any]) -> str:
    """Generate optimized query using LLM"""
```

### Report Generation Functions
```python
def generate_comprehensive_optimization_report(query_id: str, optimized_result: str, metrics: Dict[str, Any], analysis_result: str = "", performance_comparison: Dict[str, Any] = None, best_attempt_num: int = 1) -> str:
    """Generate comprehensive optimization report"""

def refine_report_content_with_llm(report_content: str) -> str:
    """Refine report using LLM"""
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
- Identify slow queries
- Pinpoint performance bottlenecks
- Get specific optimization recommendations

### 2. **Liquid Clustering Optimization**
- Analyze current clustering efficiency
- Get clustering key recommendations
- Generate implementation SQL

### 3. **Cost Optimization**
- Reduce data scan costs
- Optimize compute resource usage
- Improve overall query efficiency

### 4. **Performance Monitoring**
- Regular performance health checks
- Query improvement benchmarks
- Track optimization progress

## 🛠️ Troubleshooting

### Common Issues

**1. File Not Found**
```
❌ Error: Profile file not found
Solution: Verify file path and ensure JSON file exists
```

**2. LLM Configuration Error**
```
❌ Error: LLM provider not configured
Solution: Set valid API key and endpoint configuration
```

**3. Memory Issues**
```
❌ Error: Analysis failure due to large file size
Solution: Enable DEBUG_ENABLED='N' to reduce memory usage
```

### Debug Mode
Enable debug mode to retain intermediate files:
```python
DEBUG_ENABLED = 'Y'
```

This retains:
- Raw analysis data
- Intermediate calculation results
- LLM response details
- Error logs

## 🔧 Advanced Usage

### Switching Multiple LLM Providers
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
    # Execute analysis...
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Databricks Team**: For excellent SQL profiling capabilities
- **LLM Providers**: OpenAI, Anthropic, Azure for powerful AI capabilities
- **Community Contributors**: For feedback and improvements

## 📞 Support

For issues or questions:
- **GitHub Issues**: [Create an issue](https://github.com/MitsuhiroItagaki/test03/issues)
- **Documentation**: Check this README and code comments
- **Examples**: Review sample output files in the repository

---

**Created with ❤️ for the Databricks Community**