# Automation QnA Attribute Generation System

A comprehensive system for generating detailed Q&A reports for lipstick products using AI, with automatic data ingestion into Pinecone vector database for intelligent search and retrieval.

## 🚀 Overview

This system automates the creation of detailed product Q&A reports by:
- Reading product data from CSV or Excel (SharePoint) sources
- Generating comprehensive Q&A content using Claude AI with web search
- Storing results in Pinecone vector database for intelligent retrieval
- Providing search and answer capabilities for product attributes

## 📁 Project Structure

```
Automation_QnA_Attribute/
├── QnA_Generation/
│   ├── Qna_Core/
│   │   ├── QnA_main.py              # Main Q&A generation script
│   │   └── QnA_main_testing.py     # Testing version
│   └── data/
│       ├── QnA_prompt.json          # Search prompt configuration
│       ├── lipstick-qa-prompt-builder.json  # Q&A generation prompt
│       ├── lipstick_format.json     # Output format specification
│       └── lipstick_list.csv        # Local product data
├── clustering_Pinecone/
│   └── pinecone_core/
│       └── clustering_upserting.py  # Pinecone ingestion script
├── attribute_generation_upsert/
│   └── core/
│       ├── search_and_answer.py     # Search and retrieval system
│       └── lipstick_list.csv        # Product data for search
├── QnA_Generation/output/           # Generated Q&A JSON files
├── with batch/                      # Batch processing results
├── without Batch/                   # Single processing results
└── requirements.txt                 # Python dependencies
```

## 🛠️ Components

### 1. Q&A Generation (`QnA_Generation/`)
- **Main Script**: `QnA_main.py` - Generates detailed Q&A reports using Claude AI
- **Features**:
  - Supports both CSV and Excel (SharePoint) data sources
  - Batch processing with Anthropic Message Batches API
  - Checkpoint system for resume capability
  - Automatic Pinecone ingestion
  - Comprehensive error handling and logging

### 2. Pinecone Integration (`clustering_Pinecone/`)
- **Ingestion Script**: `clustering_upserting.py` - Uploads Q&A data to Pinecone
- **Features**:
  - Automatic index creation
  - Embedding generation with OpenAI
  - Structured metadata storage
  - Confidence score indexing

### 3. Search & Retrieval (`attribute_generation_upsert/`)
- **Search Engine**: `search_and_answer.py` - Intelligent product search
- **Features**:
  - Vector similarity search
  - Metadata filtering by SKU, category, etc.
  - Claude-powered answer generation
  - Identity-based query building

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- API Keys for:
  - Anthropic (Claude AI)
  - OpenAI (Embeddings)
  - Pinecone (Vector Database)

### Installation

1. **Clone and navigate to project**:
```bash
cd /home/sid/Documents/Automation_QnA_Attribute
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set environment variables**:
```bash
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENAI_API_KEY="your_openai_key"
export PINECONE_API_KEY="your_pinecone_key"
```

## 🚀 Usage

### Generate Q&A Reports

**To generate Q&As, run this file**:
```bash
python QnA_Generation/Qna_Core/QnA_main.py
```

**Output will be visible here**: `/home/sid/Documents/Automation_QnA_Attribute/QnA_Generation/output`

**Basic usage** (CSV data source):
```bash
python QnA_Generation/Qna_Core/QnA_main.py
```

**With Excel data source**:
1. Set `USE_EXCEL_DATA = True` in `QnA_main.py`
2. Update SharePoint URL in the script
3. Run the same command

**Options**:
- `--debug`: Enable detailed debug output
- `--no_batch`: Use synchronous processing instead of batches
- `--no_ingest`: Skip automatic Pinecone ingestion
- `--no_cache`: Disable prompt caching

### Search Product Attributes

**Direct search**:
```bash
python attribute_generation_upsert/core/search_and_answer.py
```

**Programmatic usage**:
```python
from attribute_generation_upsert.core.search_and_answer import answer_with_filters

# Search with filters
result = answer_with_filters(
    query="What is the longevity of this lipstick?",
    sku="1LYl7iHnvGklgFxvUrrLx8",
    category="Makeup"
)
print(result)
```

### Manual Pinecone Ingestion

```bash
export FILE_PATH="/path/to/your/qna_output.json"
python clustering_Pinecone/pinecone_core/clustering_upserting.py
```

## 📊 Data Sources

### CSV Format
Required columns (case-insensitive):
- `Brand` - Product brand name
- `Product_name` - Product line name
- `Shade` - Shade/color name
- `Kult SKU Code` - Unique product identifier

Optional columns:
- `Category`, `Sub Category`, `Sub sub category`
- Color metrics: `L*`, `a*`, `b*`, `C*`, `h°`, `sR`, `sG`, `sB`, `Gloss`

### Excel (SharePoint) Format
Same column structure as CSV, accessible via SharePoint URL.

## 🔍 Output Format

Generated Q&A files follow this structure:
```json
{
  "product": {
    "brand": "Colorbar",
    "product_line": "Matte Me As I Am Lipcolor",
    "shade": "002 Sabotage",
    "full_name": "Colorbar Matte Me As I Am Lipcolor 002 Sabotage",
    "sku": "1LYl7iHnvGklgFxvUrrLx8",
    "category": "Makeup",
    "sub_category": "Lip",
    "leaf_level_category": "Lipstick"
  },
  "sections": [
    {
      "title": "Section Title",
      "qas": [
        {
          "q": "Question text",
          "a": "Detailed answer",
          "why": "Scientific explanation",
          "solution": "Practical solution",
          "CONFIDENCE": "High | Source: Review consensus | Context: Based on 50+ reviews"
        }
      ]
    }
  ]
}
```

## 🎯 Key Features

### Checkpoint System
- Automatic progress tracking
- Resume interrupted processing
- Skip completed products
- Failure tracking and retry capability

### Confidence Scoring
Each Q&A includes confidence levels:
- **High**: Strong review consensus or verified data
- **Medium**: Expert analysis with limited data
- **Low**: Speculation or unverified claims

### Intelligent Search
- Vector similarity matching
- Metadata filtering
- Identity-based queries
- Context-aware answers

## 📈 Performance

### Token Usage Optimization
- Prompt caching for batch processing
- Efficient embedding generation
- Minimal API calls through batching

### Processing Speed
- Batch processing: ~4 products per batch
- Average: 30-60 seconds per product
- Checkpoint resume for large datasets

## 🔧 Configuration

### Environment Variables
```bash
# Required
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key  
PINECONE_API_KEY=your_key

# Optional
PINECONE_INDEX_NAME=qna-attributes
PINECONE_NAMESPACE=default
PINECONE_ENVIRONMENT=us-east-1
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
BATCH_SIZE=100
```

### Script Configuration
Key settings in `QnA_main.py`:
```python
USE_EXCEL_DATA = False          # True for SharePoint Excel
WRITE_RAW_FULL = True          # Save raw AI responses
WRITE_AUDIT_FILES = False      # Save processing audit logs
BATCH_SIZE_DEFAULT = 4         # Batch processing size
DEFAULT_TEMPERATURE = 0.5      # AI creativity level
```

## 📝 Logging

Comprehensive logging includes:
- Processing progress and timing
- Token usage tracking
- Error details and recovery
- Cache performance metrics
- Ingestion status

Logs are saved to: `QnA_Generation/output/logs/`

## 🚨 Error Handling

### Automatic Recovery
- JSON parsing errors → Raw response saved for manual review
- API failures → Checkpoint tracking for retry
- Network issues → Exponential backoff (planned)

### Manual Recovery
- Check `.raw.json` files for failed parsing
- Review checkpoint status
- Restart with existing checkpoint

## 🤝 Contributing

1. Follow existing code structure
2. Add comprehensive logging
3. Update checkpoint system for new features
4. Test with both CSV and Excel data sources
5. Document configuration changes

## 📄 License

Internal project - All rights reserved.

## 🆘 Support

For issues:
1. Check logs in `output/logs/`
2. Review checkpoint status
3. Verify API key configuration
4. Check data source accessibility

---

**Last Updated**: October 2025
**Version**: 2.0
**Maintainer**: Sid
