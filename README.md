# EDA AI Agent

An AI-powered Exploratory Data Analysis platform that automates data profiling, quality assessment, preprocessing, visualization, machine learning, and intelligent dataset question answering.

## Overview

EDA AI Agent is a Python-based application designed to simplify the data analysis workflow. It combines traditional statistical and machine learning techniques with Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

The platform helps users understand datasets, identify data quality issues, prepare data, train machine learning models, and ask natural-language questions about their data.

## Key Features

### 1. Automated Exploratory Data Analysis
- Dataset profiling and column summaries.
- Numerical and categorical analysis.
- Data type and column-role diagnostics.
- Missing-value and duplicate detection.
- Outlier assessment.

### 2. Data Cleaning and Preprocessing
- Missing-value handling.
- Duplicate removal.
- Categorical encoding.
- Numerical scaling.
- Datetime feature extraction.
- Processed dataset export.

### 3. Interactive Visualizations
- Histograms.
- Boxplots.
- Correlation heatmaps.
- Scatter plots.
- Categorical bar charts.

### 4. Machine Learning
- Classification.
- Regression.
- Clustering.
- Anomaly detection.
- Model evaluation and comparison.
- Cross-validation and learning-curve analysis.
- PyTorch neural network integration.

### 5. AI-Powered Dataset Question Answering
- Natural-language dataset questions.
- Hugging Face sentence embeddings.
- LangChain orchestration.
- ChromaDB vector storage and retrieval.
- Ollama local language model integration.
- Python-based evidence verification.

### 6. Backend API
FastAPI provides endpoints for dataset uploads, profiling, quality assessment, visualizations, preprocessing, machine learning, and RAG-based question answering.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Machine learning |
| PyTorch | Neural networks |
| Plotly | Interactive visualizations |
| Streamlit | User interface |
| FastAPI | Backend API |
| Hugging Face | Text embeddings |
| LangChain | LLM orchestration |
| ChromaDB | Vector database |
| Ollama | Local LLM execution |

## Project Structure

```text
eda-ai-agent/
│
├── modules/
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── data_quality.py
│   ├── visualizer.py
│   ├── preprocessor.py
│   ├── insights.py
│   ├── classification.py
│   ├── regression.py
│   ├── ml_clustering.py
│   ├── anomaly_detection.py
│   ├── pytorch_model.py
│   ├── llm_engine.py
│   ├── rag_engine.py
│   ├── rag_qa.py
│   └── langchain_engine.py
│
├── backend/
│   └── main.py
│
├── datasets/
├── reports/
├── tests/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd eda-ai-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Ollama

Install Ollama and download the model:

```bash
ollama pull llama3.2:1b
```

Ensure Ollama is running before using local LLM features.

## Running the Application

### Streamlit interface

```bash
python -m streamlit run app.py
```

### FastAPI backend

```bash
python -m uvicorn backend.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## Running Tests

Run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

**Current test status: 23 tests passing.**

## AI Architecture

The dataset question-answering workflow combines:

1. Python-generated dataset insights.
2. Hugging Face text embeddings.
3. ChromaDB semantic retrieval.
4. LangChain orchestration.
5. Ollama language model responses.
6. Python-based verification for supported categorical results.

## Future Improvements

- Expand automated test coverage.
- Improve LLM response validation.
- Add deployment configuration.
- Improve documentation and user experience.
- Explore additional Hugging Face models.

## Project Status

The project includes working EDA, preprocessing, machine learning, RAG, LangChain integration, and FastAPI functionality.

Further testing, documentation, and deployment preparation are ongoing.

## Author

Developed as a portfolio project demonstrating the integration of data science, machine learning, and AI technologies.