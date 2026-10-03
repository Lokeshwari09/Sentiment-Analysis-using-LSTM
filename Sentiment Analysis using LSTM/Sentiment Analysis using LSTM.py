import os
import re
import numpy as np
VOCAB_SIZE = 10000        
MAX_LEN = 200             
EMBED_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 10
MODEL_PATH = "sentiment_lstm.keras"
HISTORY_PATH = "training_history.png"

def train():
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.utils import pad_sequences
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.metrics import classification_report, confusion_matrix
 
    print("Loading IMDB dataset (downloads automatically on first run)...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    print(f"Train reviews: {len(x_train)} | Test reviews: {len(x_test)}")
    print("Labels: 0 = Negative, 1 = Positive")
 
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
 
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(VOCAB_SIZE, EMBED_DIM),
        layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
 
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
    ]
 
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
 
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc * 100:.2f}% | Test loss: {loss:.4f}")
 
    y_pred = (model.predict(x_test, verbose=0) > 0.5).astype(int).ravel()
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='train_acc')
    axes[0].plot(history.history['val_accuracy'], label='val_acc')
    axes[0].set_title('Accuracy'); axes[0].legend()
    axes[1].plot(history.history['loss'], label='train_loss')
    axes[1].plot(history.history['val_loss'], label='val_loss')
    axes[1].set_title('Loss'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(HISTORY_PATH)
    print(f"\nTraining done! Model saved to '{MODEL_PATH}', graph saved to '{HISTORY_PATH}'.")
 
 
def encode_text(text, word_index):
    """Convert raw text into the same integer format used by the IMDB dataset."""
    from tensorflow.keras.utils import pad_sequences
 
    text = re.sub(r"<.*?>", " ", text.lower())          # remove html tags
    words = re.findall(r"[a-z']+", text)
    seq = [1]                                            # 1 = start token
    for w in words:
        idx = word_index.get(w)
        if idx is None:
            seq.append(2)                                # 2 = unknown word
        else:
            idx += 3                                     # IMDB offset
            seq.append(idx if idx < VOCAB_SIZE else 2)
    return pad_sequences([seq], maxlen=MAX_LEN)
 
 
def predict_loop():
    from tensorflow.keras.models import load_model
    from tensorflow.keras.datasets import imdb
 
    print("Loading model...")
    model = load_model(MODEL_PATH)
    word_index = imdb.get_word_index()
 
    print("\nModel ready. Type a movie review to find its sentiment.")
    print("Type 'exit' to stop.\n")
 
    while True:
        text = input("Enter review: ").strip()
        if text.lower() == 'exit':
            print("Done.")
            break
        if text == "":
            continue
        score = float(model.predict(encode_text(text, word_index), verbose=0)[0][0])
        label = "Positive" if score >= 0.5 else "Negative"
        conf = score if score >= 0.5 else 1 - score
        print(f"\n>>> Sentiment: {label} ({conf * 100:.1f}% confidence)\n")
 
 
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model found at '{MODEL_PATH}'. Training now...\n")
        train()
    predict_loop()
