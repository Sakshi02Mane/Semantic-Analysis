# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')

# ---------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------
df = pd.read_excel("newdataset.xlsx")
df.columns = df.columns.str.strip()

# Fix sentiment values if they are strings
df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
df['sentiment'] = df['sentiment'].replace({
    'positive': 1,
    'pos': 1, 'p': 1, '1': 1,
    'negative': 0,
    'neg': 0, 'n': 0, '0': 0
})
df['sentiment'] = df['sentiment'].astype(int)

texts = df['review'].astype(str).values
labels = df['sentiment'].values

# ---------------------------------------------------
# 2. NLTK PREPROCESSING FUNCTION
# ---------------------------------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

processed_texts = [preprocess(t) for t in texts]

# ---------------------------------------------------
# 3. TF-IDF ENCODING
# ---------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(processed_texts)
y = labels

# ---------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

# ---------------------------------------------------
# 5. TRAIN MODEL FOR MULTI-EPOCH ACCURACY GRAPH
# ---------------------------------------------------
model = LogisticRegression(max_iter=1000)

epochs = 10
train_acc = []
val_acc = []

for epoch in range(epochs):
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    val_accuracy = model.score(X_test, y_test)

    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)

    print(f"Epoch {epoch+1}/{epochs} => Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}")

# ---------------------------------------------------
# 6. ACCURACY GRAPH
# ---------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title("NLTK + Logistic Regression Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("nltk_accuracy_graph.png")
print("✅ Graph saved as nltk_accuracy_graph.png")

# ---------------------------------------------------
# 7. PREDICTION FUNCTION
# ---------------------------------------------------
def predict_sentiment(text):
    clean = preprocess(text)
    vector = vectorizer.transform([clean])
    pred = model.predict(vector)[0]
    
    if pred == 1:
        return "Positive 👍"
    else:
        return "Negative 👎"

# ---------------------------------------------------
# TEST
# ---------------------------------------------------
print(predict_sentiment("This product was really good!"))





