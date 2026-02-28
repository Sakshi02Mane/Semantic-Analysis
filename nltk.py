import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

DATA_PATH       = "newdataset.xlsx"
MAX_FEATURES    = 5000
TEST_SIZE       = 0.2
RANDOM_SEED     = 42
MODEL_PATH      = "nltk_logistic_model.pkl"
VECTORIZER_PATH = "nltk_tfidf_vectorizer.pkl"

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()

df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
df['sentiment'] = df['sentiment'].replace({
    'positive': 'positive', 'pos': 'positive', 'p': 'positive', '1': 'positive',
    'neutral':  'neutral',  'neu': 'neutral',
    'negative': 'negative', 'neg': 'negative', 'n': 'negative', '0': 'negative'
})

df = df[df['sentiment'].isin(['positive', 'neutral', 'negative'])]
df = df[['review', 'sentiment']].dropna()
df['review'] = df['review'].astype(str)

texts  = df['review'].values
labels = df['sentiment'].values

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

print("Preprocessing texts...")
processed_texts = [preprocess(t) for t in texts]

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(processed_texts)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

model  = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
EPOCHS = 10
train_acc_list = []
val_acc_list   = []

for epoch in range(EPOCHS):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred   = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc   = accuracy_score(y_test,  val_pred)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

print("\nFinal Accuracy:", val_acc_list[-1])
print("Final F1 (macro):", f1_score(y_test, val_pred, average="macro"))
print("\nClassification Report:\n", classification_report(y_test, val_pred))

joblib.dump(model,      MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print("Model saved as", MODEL_PATH)
print("Vectorizer saved as", VECTORIZER_PATH)

plt.figure(figsize=(8, 5))
plt.plot(train_acc_list, label='Training Accuracy')
plt.plot(val_acc_list,   label='Validation Accuracy')
plt.title("NLTK + Logistic Regression Sentiment Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("nltk_accuracy_graph.png")
print("Accuracy graph saved as nltk_accuracy_graph.png")

def predict_sentiment(text):
    clean  = preprocess(text)
    vector = vectorizer.transform([clean])
    pred   = model.predict(vector)[0]
    return pred.capitalize()

if __name__ == "__main__":
    print(predict_sentiment("This product is absolutely amazing!"))
    print(predict_sentiment("It was okay, nothing special."))
    print(predict_sentiment("Worst experience ever, very disappointed."))
    