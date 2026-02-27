# 🧠 Semantic Analysis — Hybrid Sentiment Classification

A hybrid machine learning project that classifies text reviews as **Positive**, **Negative**, or **Neutral** using three different approaches — compared side by side to evaluate performance.

> Built for Girl Hackathon Silicon 2025.

---

## 📁 Project Structure

```
Semantic-Analysis/
├── cnn_lstm.py                      # Deep learning model (CNN + LSTM)
├── logistic_regression.py           # Baseline TF-IDF + Logistic Regression
├── nltk_logistic.py                 # NLTK preprocessing + Logistic Regression
├── newdataset.xlsx                  # Dataset (reviews + sentiment labels)
├── logistic_sentiment_model.pkl     # Saved logistic regression model
├── tfidf_vectorizer.pkl             # Saved TF-IDF vectorizer
├── accuracy_graph.png               # CNN+LSTM accuracy curve
├── nltk_accuracy_graph.png          # NLTK model accuracy curve
└── sentiment_accuracy_graph.png     # Logistic regression accuracy curve
```

---

## ⚙️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data loading and manipulation |
| NLTK | Text preprocessing (tokenization, stemming, stopword removal) |
| Scikit-learn | TF-IDF vectorization, Logistic Regression, metrics |
| TensorFlow / Keras | CNN + LSTM deep learning model |
| Matplotlib | Accuracy curve visualization |
| Joblib | Saving and loading trained models |

---

## 🔬 Three Approaches Compared

### 1. Baseline — TF-IDF + Logistic Regression (`logistic_regression.py`)
The simplest approach. Converts raw text into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency) with bigrams, then trains a Logistic Regression classifier.

- No text preprocessing
- TF-IDF with `max_features=5000`, `ngram_range=(1,2)`
- Fast to train, strong baseline

### 2. NLTK + Logistic Regression (`nltk_logistic.py`)
Same classifier but with proper NLP preprocessing applied first — making the features much cleaner before vectorization.

**Preprocessing pipeline:**
- Lowercasing
- Tokenization via NLTK `word_tokenize`
- Stopword removal (removes "the", "is", "and" etc.)
- Porter Stemming (reduces "running" → "run", "better" → "better")

- TF-IDF with `max_features=5000`
- Better generalization than the baseline due to cleaner input

### 3. CNN + LSTM (`cnn_lstm.py`)
A deep learning model combining Convolutional Neural Networks and Long Short-Term Memory networks — the most powerful approach in this project.

**Architecture:**
```
Embedding (10000 vocab, 64 dims)
      ↓
Conv1D (64 filters, kernel=3, ReLU)   ← extracts local patterns
      ↓
LSTM (64 units)                        ← captures sequence context
      ↓
Dense (32, ReLU)
      ↓
Dropout (0.3)                          ← prevents overfitting
      ↓
Dense (1, Sigmoid)                     ← binary output
```

- Learns directly from raw token sequences
- No manual feature engineering needed
- Captures both local word patterns (CNN) and long-range dependencies (LSTM)

---

## 🧹 Data Preprocessing

All three models handle inconsistent label formats in the dataset:

```python
# Maps all label variations to standard form
'positive', 'pos', 'p', '1'  →  positive (1)
'negative', 'neg', 'n', '0'  →  negative (0)
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn tensorflow nltk matplotlib openpyxl joblib
```

### Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Run Any Model

```bash
# Baseline Logistic Regression
python logistic_regression.py

# NLTK + Logistic Regression
python nltk_logistic.py

# CNN + LSTM Deep Learning
python cnn_lstm.py
```

---

## 📡 Predict on New Text

**Logistic Regression:**
```python
from logistic_regression import predict_sentiment
print(predict_sentiment("This product is amazing!"))
# → Positive
```

**CNN + LSTM:**
```python
print(predict_sentiment("Terrible experience, never buying again"))
# → Negative 👎 (score=0.023)
```

---

## 📊 Dataset

- File: `newdataset.xlsx`
- Columns: `review` (text), `sentiment` (label)
- Labels: positive / negative / neutral
- 80% training / 20% validation split

---

## 🧠 Key Concepts Demonstrated

- **TF-IDF Vectorization** — converts text to numerical features based on word importance
- **Bigrams** — captures two-word phrases like "not good" as single features
- **Stemming** — reduces words to their root form for better generalization
- **Stopword Removal** — filters out common words that carry no sentiment signal
- **CNN for NLP** — extracts local patterns within a window of words
- **LSTM** — remembers context across long sequences, handles word order
- **Dropout Regularization** — randomly disables neurons during training to prevent overfitting
- **Model Persistence** — saves trained models with joblib for reuse without retraining

---

## 👩‍💻 Author

**Sakshi Mane** — [GitHub](https://github.com/Sakshi02Mane)
