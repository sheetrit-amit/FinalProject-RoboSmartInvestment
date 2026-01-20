# 🤖 RoboSmartInvestment
# Project Demo- https://drive.google.com/file/d/1Ukn3Zli8aTwzwgCfSZPyXSiDvDxNIDzA/view
**AI-Powered Smart Stock Portfolio Builder**

An intelligent investment system that combines machine learning, modern portfolio theory, and LLM-based analysis to create personalized stock portfolios tailored to individual risk preferences.

---

## 🎯 Project Overview

RoboSmartInvestment is an end-to-end automated system that helps investors build optimized stock portfolios by:

1. **Understanding investor preferences** - Risk tolerance and investment amount
2. **Classifying stocks by risk** - Using Decision Tree machine learning
3. **Optimizing portfolio allocation** - Using Markowitz Modern Portfolio Theory
4. **Analyzing fundamentals with AI** - LLM-powered quarterly report analysis
5. **Generating personalized recommendations** - Final LLM synthesis with explanations

---

## 🔄 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
│                    💰 Investment Amount + 📊 Risk Tolerance                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RISK CLASSIFICATION                                  │
│                    🌳 Decision Tree Classifier                               │
│         Classifies 1000+ stocks into risk levels (Low/Medium/High)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   Stocks matching user's risk level │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌───────────────────────────────┐     ┌───────────────────────────────────────┐
│     MARKOWITZ OPTIMIZATION    │     │         N8N + LLM ANALYSIS            │
│   📈 Modern Portfolio Theory  │     │   📄 Quarterly Reports Processing     │
│                               │     │                                       │
│ • Expected returns            │     │ • Fetch latest quarterly reports      │
│ • Risk (volatility)           │     │ • LLM analyzes financial health       │
│ • Correlation matrix          │     │ • Generate fundamental score          │
│ • Efficient frontier          │     │                                       │
│ • Optimal weights             │     │                                       │
└───────────────────────────────┘     └───────────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FINAL LLM SYNTHESIS                                 │
│                      🧠 Portfolio Construction AI                            │
│                                                                              │
│  Combines:                                                                   │
│  • Markowitz optimal weights                                                 │
│  • LLM fundamental scores                                                    │
│  • User risk preferences                                                     │
│                                                                              │
│  Outputs:                                                                    │
│  ✅ Final portfolio allocation                                               │
│  ✅ Investment reasoning & interpretation                                    │
│  ✅ Risk assessment & recommendations                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Data Storage** | Google BigQuery |
| **Data Source** | Yahoo Finance API |
| **ML Classification** | Decision Tree (scikit-learn) |
| **Portfolio Optimization** | Markowitz Model (Python) |
| **Workflow Automation** | n8n |
| **AI Analysis** | Large Language Models (LLM) |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
FinalProject-RoboSmartInvestment/
│
├── data/
│   ├── tickers_top1000.txt           # Main stock universe (1000 companies)
│   ├── tickers_training_200.txt      # Training set for decision tree
│   ├── ticker_sector_training.csv    # Sector classification
│   └── raw/                          # Raw data files
│
├── src/
│   └── data_retrieval/
│       ├── yahoo_to_bigquery.py              # Daily prices → BigQuery
│       ├── bulk_load_to_bigquery.py          # Bulk loading utility
│       ├── financial_statements_to_bigquery.py # Financial statements
│       ├── upload_training_data_to_bigquery.py # Training data upload
│       ├── get_ticker_sectors.py             # Sector classification
│       └── check_missing_tickers.py          # Data validation
│
├── notebooks/
│   └── data_analysis_eda.ipynb       # Exploratory data analysis
│
├── docs/
│   └── data_analysis.md              # Data analysis documentation
│
└── requirements.txt                  # Python dependencies
```

---

## 📊 Data Pipeline

### BigQuery Datasets

**`StockData`** - Main dataset (1000 companies)
| Table | Description |
|-------|-------------|
| `daily_prices` | OHLCV data (5 years) |
| `income_statements` | Revenue, expenses, profits |
| `balance_sheets` | Assets, liabilities, equity |
| `cash_flows` | Operating, investing, financing |

**`DecisionTreeTraining`** - Training dataset (200 companies)
| Table | Description |
|-------|-------------|
| `daily_prices` | Training price data |
| `income_statements` | Training income data |
| `balance_sheets` | Training balance data |
| `cash_flows` | Training cash flow data |
| `ticker_sectors` | Sector classification |

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# Google Cloud SDK configured
gcloud auth application-default login
```

### Installation

```bash
# Clone the repository
git clone git@github.com:sheetrit-amit/FinalProject-RoboSmartInvestment.git
cd FinalProject-RoboSmartInvestment

# Install dependencies
pip install -r requirements.txt
```

### Load Data to BigQuery

```bash
# 1. Load main stock universe (1000 companies)
python src/data_retrieval/bulk_load_to_bigquery.py

# 2. Load financial statements
python src/data_retrieval/financial_statements_to_bigquery.py

# 3. Load training data (200 companies)
python src/data_retrieval/upload_training_data_to_bigquery.py

# 4. Generate sector classifications
python src/data_retrieval/get_ticker_sectors.py
```

---

## 📈 Key Features

### 1. Risk Classification (Decision Tree)
- Trained on 200 diverse companies
- Features: volatility, beta, financial ratios
- Output: Low / Medium / High risk classification

### 2. Markowitz Portfolio Optimization
- Calculates expected returns and covariance
- Generates efficient frontier
- Finds optimal portfolio weights for target risk

### 3. LLM Financial Analysis
- Processes quarterly earnings reports
- Analyzes management commentary
- Generates fundamental health scores

### 4. AI Portfolio Synthesis
- Combines quantitative and qualitative analysis
- Generates human-readable investment rationale
- Provides actionable portfolio recommendations

---

## 🎓 Academic Context

This project is developed as a final year project at Ben-Gurion University of the Negev (BGU), combining:
- **Machine Learning** - Classification algorithms
- **Financial Theory** - Modern Portfolio Theory
- **Natural Language Processing** - LLM analysis
- **Data Engineering** - BigQuery, ETL pipelines

---

## 📝 License

This project is for academic purposes.

---

## 👤 Author

**Amit Sheetrit**
- GitHub: [@sheetrit-amit](https://github.com/sheetrit-amit)
- University: Ben-Gurion University of the Negev

---

<p align="center">
  <i>Building smarter investment decisions with AI 🚀</i>
</p>
