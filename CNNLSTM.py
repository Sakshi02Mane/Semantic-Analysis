# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense, Dropout

# ---------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------
df = pd.read_excel("newdataset.xlsx")
df.columns = df.columns.str.strip()

# ✅ Clean and convert sentiment column
df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()

df['sentiment'] = df['sentiment'].replace({
    'positive': 1,
    'pos': 1,
    'p': 1,
    '1': 1,
    'negative': 0,
    'neg': 0,
    'n': 0,
    '0': 0
})

df['sentiment'] = df['sentiment'].astype(int)

texts = df["review"].astype(str).values
labels = df["sentiment"].values

# ---------------------------------------------------
# 2. TOKENIZATION
# ---------------------------------------------------
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_len = 100
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# ---------------------------------------------------
# 3. CNN + LSTM MODEL
# ---------------------------------------------------
model = Sequential([
    Embedding(10000, 64, input_length=max_len),

    Conv1D(filters=64, kernel_size=3, activation='relu'),

    LSTM(64),

    Dense(32, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------
history = model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)

# ---------------------------------------------------
# 5. SAVE ACCURACY GRAPH
# ---------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CNN + LSTM Sentiment Model Accuracy")
plt.legend()
plt.savefig("accuracy_graph.png")
print("✅ Saved accuracy graph as accuracy_graph.png")

# ---------------------------------------------------
# 6. PREDICTOR
# ---------------------------------------------------
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0][0]

    if pred > 0.5:
        return f"Positive 👍 (score={pred:.3f})"
    else:
        return f"Negative 👎 (score={pred:.3f})"

# ---------------------------------------------------
# TEST
# ---------------------------------------------------
print(predict_sentiment("This product is amazing"))

