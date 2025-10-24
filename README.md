# Spam SMS Classification

This project classifies SMS messages as **spam** or **ham** using traditional ML and deep learning methods, including embeddings, LSTM, and transfer learning with the **Universal Sentence Encoder (USE)**.

---

## Dataset
- SMS messages labeled `spam` or `ham`.  
- Preprocessed columns: `label`, `Text`, `label_enc` (`0`=ham, `1`=spam).  

---

## Preprocessing
- Dropped unnecessary unnamed columns.  
- Renamed `v1` → `label`, `v2` → `Text`.  
- Encoded labels numerically.  
- Avg words/message: 15 | Unique words: 15,585  
- Split: 80% train, 20% test.

---

## Baseline Model
- **MultinomialNB** with TF-IDF vectors.  
- **Accuracy:** 96.2% | **F1 (spam):** 0.837  

---

## Deep Learning Models

### 1. Custom Text Vectorization & Embeddings
- TensorFlow `TextVectorization` + `Embedding(128)`  
- GlobalAveragePooling + Dense layers  
- **Accuracy:** 98.3% | **F1 (spam):** 0.934  

### 2. Bidirectional LSTM
- Bi-LSTM → LSTM → Dense + Dropout  
- **Accuracy:** 98.6% | **F1 (spam):** 0.944  

### 3. Transfer Learning (USE)
- **USE embeddings** from TF Hub + Dense layers  
- **Accuracy:** 98.2% | **F1 (spam):** 0.931  

---

## Model Evaluation

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MultinomialNB | 0.962 | 1.0 | 0.72 | 0.837 |
| Custom Embedding | 0.983 | 0.978 | 0.893 | 0.934 |
| Bi-LSTM | 0.986 | 0.993 | 0.90 | 0.944 |
| USE Transfer | 0.982 | 0.964 | 0.90 | 0.931 |

---

## Conclusion
- Bi-LSTM achieves highest accuracy & F1.  
- Deep learning with embeddings outperforms traditional ML.  
- USE transfer learning is fast and effective.  

---

## Requirements
numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow, tensorflow-hub


---

## How to Run
1. Clone repo.  
2. Install dependencies.  
3. Place `spam.csv` in working directory.  
4. Run notebook/script to train & evaluate models.
