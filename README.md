# Classifying Banking Intent from Customer Queries

An end-to-end NLP data science portfolio project focused on automatically classifying customer banking queries into 77 distinct intent categories using PyTorch, Hugging Face Transformers, and LoRA (Low-Rank Adaptation).

## Project Overview
This project implements the **CRISP-DM** framework to build, train, evaluate, and deploy a robust text classification pipeline. It compares a traditional non-contextual baseline (TF-IDF + Multi-Layer Perceptron) against an advanced context-aware transformer model (**RoBERTa-base** fine-tuned with PEFT/LoRA).

## Project Structure
```text
Classifying-Banking-Intent/
│
├── datasets/
│   ├── banking77_train.csv
│   └── banking77_test.csv
│
├── banking_classification_BERT.ipynb
├── banking_classification_BERT_solution.ipynb
├── requirements.txt
└── README.md

```

## Key Results & Performance

* **Baseline MLP (TF-IDF):** Fast to train, serving as a reliable non-contextual benchmark.
* **LoRA-RoBERTa (Advanced):** Achieved **~90.4% Overall Accuracy** and **~90.4% Macro F1-Score**, training only **~1.21%** of the total parameters thanks to parameter-efficient fine-tuning.

## AI Ethics & Data Privacy

The inference pipeline includes an automated **PII (Personally Identifiable Information) Redaction Module** using regular expressions to sanitize sensitive details (such as credit card numbers, email addresses, and phone numbers) prior to model ingestion, ensuring strict data security compliance.

## Conclusions and Final Findings

* **What are the key findings and takeaways?** Moving from a traditional non-contextual baseline (TF-IDF + MLP) to a state-of-the-art transformer architecture using Parameter-Efficient Fine-Tuning (LoRA) highlighted the importance of managing tensor dimensions, resolving library dependencies (like `torchao`), and integrating robust PII masking for enterprise-grade NLP deployments.
* **What did you learn through the process?** We learned how to scale text classification effectively while fine-tuning only ~1.2% of total parameters, making state-of-the-art transformer performance exceptionally fast and resource-efficient on a standard GPU.
* **Are the results what you expected?** Yes. While the baseline MLP performed reasonably well given the concise nature of the queries, fine-tuning RoBERTa with LoRA pushed overall accuracy and Macro F1-Score past ~90.4%, successfully capturing subtle semantic similarities across the 77 intent classes.
* **How can we improve our results?** Future improvements will focus on exploring larger transformer variants (e.g., `roberta-large`), optimizing hyperparameters and LoRA rank, and implementing confidence score thresholds for low-confidence predictions.
* **What are the next steps?** Containerizing the inference pipeline using FastAPI, adding a monitoring framework for data drift, and publishing the complete portfolio showcase on GitHub and LinkedIn.

## Setup & Installation

1. Clone the repository:
```bash
git clone git@github.com:gabrielarcangelbol/Classifying-Banking-Intent.git
cd Classifying-Banking-Intent

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Open and run `banking_classification_BERT.ipynb` in Google Colab or your local Jupyter environment.