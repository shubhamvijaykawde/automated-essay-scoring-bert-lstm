#Parameter-Based Automated Essay Grading using LSTM and BERT

This project implements an Automated Essay Grading (AEG) system using two complementary deep learning approaches:

1. LSTM + Word2Vec for holistic score prediction (ASAP dataset)
2. BERT-based multi-output regression for parameter-wise essay evaluation (CELA dataset)

The system predicts essay quality either as a single holistic score or as multiple fine-grained linguistic parameters such as grammar, lexical quality, organization, and coherence.

---

##Project Motivation

Traditional automated essay grading systems focus primarily on holistic scores, offering limited interpretability.  
This project extends standard approaches by:

- Combining sequential modeling (LSTM) with contextual embeddings (BERT)
- Supporting parameter-level scoring
- Providing a web-based interface for real-time essay evaluation

---

##Models Overview

### 1) LSTM-based Holistic Scoring
- Dataset: ASAP (Automated Student Assessment Prize)
- Embeddings: Word2Vec (300-dimensional)
- Architecture:
  - LSTM (300 units) ? LSTM (64 units) ? Dense(1)
- Evaluation Metrics:
  - RMSE
  - Quadratic Weighted Kappa (QWK)

### 2) BERT-based Parameter Scoring
- Dataset: CELA
- Model: `bert-base-uncased`
- Task: Multi-output regression
- Outputs:
  - Grammar
  - Lexical Quality
  - Global Organization
  - Local Organization
  - Supporting Ideas
  - Holistic Score
- Loss Function: Mean Squared Error (MSE)

---

