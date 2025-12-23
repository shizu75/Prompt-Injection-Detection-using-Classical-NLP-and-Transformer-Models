# Prompt Injection Detection using Classical NLP and Transformer Models

## Overview
This project presents a **robust prompt injection detection system** that combines **classical NLP baselines** with **state-of-the-art transformer models** to identify malicious and adversarial prompts. The system is designed to detect both **vanilla** and **adversarial prompt injections** in large language model (LLM) inputs.

The pipeline spans **data aggregation, preprocessing, baseline modeling, transformer fine-tuning, evaluation, and interactive deployment**, making it suitable for **security research, AI safety, and applied ML engineering**.

---

## Problem Motivation
Prompt injection attacks pose a serious threat to LLM-based systems by manipulating system instructions, bypassing safeguards, or extracting confidential information. Detecting such attacks is critical for:
- AI safety
- Secure LLM deployment
- Enterprise and regulated AI systems
- Alignment research

---

## Dataset Construction

### Sources
- **Safe-Guard Prompt Injection Dataset (HuggingFace)**
- **Malificent Prompt Dataset (CSV)**

### Label Mapping
| Category | Label |
|--------|-------|
| Vanilla Harmful | 1 |
| Adversarial Harmful | 1 |
| Vanilla Benign | 0 |
| Adversarial Benign | 0 |

### Processing Steps
- Merge multiple datasets
- Standardize column names
- Remove duplicates and null values
- Stratified splitting:
  - 70% Training
  - 15% Validation
  - 15% Testing

Final dataset saved as:

---

## Exploratory Data Analysis
- Class distribution inspection
- Null & duplicate checks
- Dataset statistics and validation
- Balanced harmful vs benign samples ensured via stratification

---

## Baseline Model: TF-IDF + Logistic Regression

### Feature Engineering
- **Character-level TF-IDF**
- N-grams: `(3, 6)`
- Max features: `50,000`

### Classifier
- Logistic Regression (`solver=saga`)
- Balanced class weights

### Metrics Evaluated
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

### Purpose
Provides a **fast, interpretable baseline** and serves as an ensemble component for deployment.

---

## Transformer Model: DistilBERT

### Architecture
- `distilbert-base-uncased`
- Sequence length: `256`
- Binary classification head

### Training Setup
- Optimizer: AdamW
- Learning rate: `2e-5`
- Epochs: `5`
- Batch size:
  - Train: `16`
  - Eval: `32`
- Metric for checkpoint selection: **F1-score**

### Tokenization
- Padding & truncation to 256 tokens
- HuggingFace `Datasets` pipeline

---

## Evaluation Metrics (Transformer)
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Precision–Recall Curve
- Training & validation loss curves

The transformer significantly outperforms the TF-IDF baseline on adversarial samples.

---

## Ensemble Strategy
An **optional ensemble** combines:
- DistilBERT probability
- TF-IDF Logistic Regression probability


Where `α` is user-controlled in the app.

---

## Interactive Application (Gradio)

### Features
- Real-time prompt analysis
- Adjustable decision threshold
- Ensemble weight control
- Token-level highlighting for interpretability
- Clear malicious vs benign classification

### Model Artifacts
- `distilbert-promptdetector/`
- `tfidf_vectorizer.joblib`
- `tfidf_clf.joblib`

---

## Deployment
- Hosted using **Gradio**
- HuggingFace Spaces compatible
- Git LFS enabled for model weights
- Secure token-based authentication

---

## Key Results
- High recall on malicious prompts
- Strong generalization to adversarial injections
- Robust performance across classical and neural models
- Explainable token-level outputs

---

## Limitations
- English-only prompts
- Static datasets
- No online continual learning

---

## Future Work
- Multilingual prompt injection detection
- Instruction hierarchy modeling
- Reinforcement learning–based defenses
- Integration with real-time LLM gateways
- Adversarial training with adaptive attacks

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- PyTorch
- HuggingFace Transformers & Datasets
- Gradio
- Matplotlib, Seaborn
- Joblib

---

## Research Relevance
This project aligns with research directions in:
- AI Safety
- LLM Security
- Adversarial NLP
- Responsible AI
- Prompt Engineering Defense

---

## Author
**Soban Saeed**  
AI Security | NLP | LLM Safety Research

---

> This project is intended strictly for academic, research, and defensive security purposes.

