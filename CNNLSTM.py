import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense, Dropout

DATA_PATH      = "newdataset.xlsx"
MAX_WORDS      = 10000
MAX_LEN        = 100
EMBEDDING_DIM  = 64
EPOCHS         = 10
BATCH_SIZE     = 32
TOKENIZER_PATH = "cnn_lstm_tokenizer.pkl"
MODEL_PATH     = "cnn_lstm_model.keras"

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()

df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
df['sentiment'] = df['sentiment'].replace({
    'positive': 2, 'pos': 2, 'p': 2, '1': 2,
    'neutral':  1, 'neu': 1,
    'negative': 0, 'neg': 0, 'n': 0, '0': 0
})

df = df[df['sentiment'].isin([0, 1, 2])]
df['sentiment'] = df['sentiment'].astype(int)

texts  = df['review'].astype(str).values
labels = df['sentiment'].values

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = np.array(labels)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved as", TOKENIZER_PATH)

model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

history = model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    shuffle=True
)

model.save(MODEL_PATH)
print("Model saved as", MODEL_PATH)

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'],     label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CNN + LSTM Sentiment Model Accuracy")
plt.legend()
plt.savefig("cnn_lstm_accuracy_graph.png")
print("Accuracy graph saved as cnn_lstm_accuracy_graph.png")

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text):
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    probs  = model.predict(padded)[0]
    label  = np.argmax(probs)
    return f"{LABEL_MAP[label]} (score={probs[label]:.3f})"

if __name__ == "__main__":
    print(predict_sentiment("This product is absolutely amazing!"))
    print(predict_sentiment("It was okay, nothing special."))
    print(predict_sentiment("Worst experience ever, very disappointed."))
    