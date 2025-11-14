import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib  
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # ensures graph gets saved


DATA_PATH = "newdataset.xlsx"
TEXT_COL = "review"
LABEL_COL = "sentiment"
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_SEED = 42

MODEL_PATH = "logistic_sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# ----------------------------------------
# 1. LOAD & CLEAN DATA
# ----------------------------------------
df = pd.read_excel(DATA_PATH)

if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
    raise ValueError(f"Expected columns '{TEXT_COL}' and '{LABEL_COL}' in dataset. Found: {df.columns.tolist()}")

df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].astype(str)

# Convert labels to simple numeric categories if needed
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().str.strip()
df[LABEL_COL] = df[LABEL_COL].replace({
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "pos": "positive", "p": "positive", "1": "positive",
    "neg": "negative", "n": "negative", "0": "negative"
})

# ----------------------------------------
# 2. SPLIT DATA
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL],
    df[LABEL_COL],
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df[LABEL_COL]
)

# ----------------------------------------
# 3. TF-IDF VECTORIZATION
# ----------------------------------------
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------------------
# 4. TRAIN MODEL + COLLECT ACCURACY PER EPOCH
# ----------------------------------------
EPOCHS = 10
train_acc_list = []
val_acc_list = []

model = LogisticRegression(max_iter=200, solver="liblinear", random_state=RANDOM_SEED)

for epoch in range(EPOCHS):
    model.fit(X_train_tfidf, y_train)

    train_pred = model.predict(X_train_tfidf)
    val_pred = model.predict(X_test_tfidf)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_test, val_pred)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} — Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

# ----------------------------------------
# 5. FINAL METRICS
# ----------------------------------------
print("\n✅ Final Accuracy:", val_acc_list[-1])
print("✅ Final F1 (macro):", f1_score(y_test, val_pred, average="macro"))
print("\nClassification Report:\n", classification_report(y_test, val_pred))

# ----------------------------------------
# 6. SAVE MODEL + VECTORIZER
# ----------------------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

# ----------------------------------------
# 7. PLOT ACCURACY CURVES
# ----------------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_acc_list, label="Training Accuracy")
plt.plot(val_acc_list, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Sentiment Model Accuracy")
plt.legend()

plt.savefig("sentiment_accuracy_graph.png")
print("\n✅ Accuracy graph saved as: sentiment_accuracy_graph.png")

# ----------------------------------------
# 8. PREDICTION FUNCTION
# ----------------------------------------
def predict_sentiment(text):
    clf = joblib.load(MODEL_PATH)
    vec = joblib.load(VECTORIZER_PATH)
    text_tfidf = vec.transform([text])
    prediction = clf.predict(text_tfidf)[0]
    return prediction

# ----------------------------------------
# 9. OPTIONAL: COMMAND-LINE INPUT
# ----------------------------------------
if __name__ == "__main__":
    user_input = input("\nEnter a review to analyze sentiment: ")
    sentiment = predict_sentiment(user_input)
    print(f"\nPredicted Sentiment: {sentiment.capitalize()}")

