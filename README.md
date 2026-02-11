# Automated Essay Scoring using LSTM & BERT

A deep learning–based Automated Essay Scoring (AES) system that evaluates essays either holistically or across multiple linguistic parameters such as grammar, vocabulary, organization, and supporting ideas.

The project combines sequential modeling (LSTM) and transformer-based contextual understanding (BERT) and provides real-time evaluation through a web application.

---

## Features

* Multi-parameter essay grading (Grammar, Lexical Quality, Organization, Supporting Ideas, Holistic Score)
* Holistic scoring using LSTM
* Parameter-level scoring using BERT multi-output regression
* Score capping to prevent unrealistic predictions
* Human-readable feedback generation for each score band
* Interactive Flask web interface for real-time evaluation

---

## Models

### BERT — Parameter Based Scoring

Dataset: CELA
Architecture: bert-base-uncased + regression head
Outputs:

* Grammar
* Lexical
* Global Organization
* Local Organization
* Supporting Ideas
* Holistic

### LSTM — Holistic Scoring

Dataset: ASAP
Embedding: Word2Vec
Evaluation Metrics:

* RMSE
* Quadratic Weighted Kappa (QWK)

---

## Project Structure

```
bert_models/        → BERT architecture & dataset loader
src/                → preprocessing & LSTM model
templates/          → web interface
notebooks/          → training notebooks
results/            → prediction outputs
data/               → datasets (not included)
models/             → trained weights (not included)
app.py              → Flask application
```

---

## Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/shubhamvijaykawde/automated-essay-scoring-bert-lstm.git
cd automated-essay-scoring-bert-lstm
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Download Datasets

Follow instructions in:

```
DATASET.md
```

### 4. Run Web App

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## Demo Workflow

1. Paste an essay into the text box
2. Click Evaluate
3. System predicts parameter-wise scores
4. Feedback is generated automatically

---

## Technical Highlights

* Multi-output regression using BERT
* Sequence modeling with LSTM
* Custom feedback engine mapped to scoring rubric
* Score normalization and capping to prevent inflated grading
* End-to-end ML deployment with Flask

---

## Future Improvements

* Attention visualization for interpretability
* Grammar error localization
* Rubric-aligned training with ranking loss
* GPU optimized inference

---

## Author

Shubham Vijay Kawde
MSc Data Science — Trier University
