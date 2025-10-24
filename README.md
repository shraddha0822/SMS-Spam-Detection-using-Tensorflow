# Spam SMS Classification with Machine Learning and Deep Learning

This project focuses on building and comparing multiple models to classify SMS messages as **spam** or **ham**. It includes traditional machine learning methods, custom embeddings with deep learning, LSTM models, and transfer learning using the **Universal Sentence Encoder (USE)**.

---

## Table of Contents

- [Dataset](#dataset)  
- [Data Preprocessing](#data-preprocessing)  
- [Exploratory Data Analysis](#exploratory-data-analysis)  
- [Baseline Model](#baseline-model)  
- [Deep Learning Models](#deep-learning-models)  
  - [Custom Text Vectorization & Embeddings](#custom-text-vectorization--embeddings)  
  - [Bidirectional LSTM](#bidirectional-lstm)  
  - [Transfer Learning with Universal Sentence Encoder](#transfer-learning-with-universal-sentence-encoder)  
- [Model Evaluation](#model-evaluation)  
- [Results](#results)  
- [Conclusion](#conclusion)  
- [Requirements](#requirements)  
- [How to Run](#how-to-run)  

---

## Dataset

The dataset contains SMS messages labeled as `spam` or `ham`. The original dataset has some unnecessary columns which are removed during preprocessing.  

**Columns after preprocessing:**

| Column      | Description                  |
|------------|------------------------------|
| label      | Target variable (`ham`/`spam`) |
| Text       | SMS message text              |
| label_enc  | Encoded target (`0` for ham, `1` for spam) |

---

## Data Preprocessing

1. Removed unnamed columns containing null values.  
2. Renamed columns `v1` → `label` and `v2` → `Text`.  
3. Encoded target variable using `.map()` (`ham` → 0, `spam` → 1).  
4. Calculated statistics:  
   - Average number of words per message: `15`  
   - Total unique words in corpus: `15,585`  
5. Split the dataset into training (80%) and testing (20%) sets.  

---

## Exploratory Data Analysis

- Visualized the distribution of `ham` and `spam` messages using a count plot:  
  ```python
  sns.countplot(x=df['label'])
  plt.show()
