# Sentiment Analysis using LSTM

A deep learning project that classifies movie reviews as **Positive** or **Negative** using a Bidirectional LSTM network built with TensorFlow/Keras. It is trained on the IMDB movie reviews dataset, and after training it opens an interactive console where you can type any review and get its predicted sentiment with a confidence score.

## Features

- Automatic download of the IMDB dataset (no manual files needed)
- Text preprocessing with word-index encoding and padding
- Bidirectional LSTM model with dropout regularization
- Early stopping and best-model checkpointing
- Evaluation with accuracy, classification report and confusion matrix
- Interactive prediction for custom reviews

## Dataset

**IMDB Movie Reviews** (built into Keras)

| Split | Reviews |
|---|---|
| Train | 25,000 |
| Test | 25,000 |

- Labels: `0` = Negative, `1` = Positive (balanced classes)
- Vocabulary: top 10,000 most frequent words
- 20% of the training data is held out for validation

## Project Structure

```
sentiment analysis lstm/
├── sentiment_lstm.py        # Training + prediction script
├── sentiment_lstm.keras     # Saved best model (generated after training)
├── training_history.png     # Accuracy/loss curves (generated)
└── README.md
```

## Model Architecture

```
Input (200 word IDs)
   ↓
Embedding (10000 words, 128 dimensions)
   ↓
Bidirectional LSTM (64 units)
   ↓
Dropout (0.5)
   ↓
Dense (32, ReLU)
   ↓
Dropout (0.3)
   ↓
Dense (1, Sigmoid)  → Positive / Negative
```

## Configuration

| Setting | Value |
|---|---|
| Vocabulary size | 10,000 |
| Max review length | 200 words |
| Embedding dimension | 128 |
| LSTM units | 64 (bidirectional) |
| Batch size | 64 |
| Max epochs | 10 |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Callbacks | ModelCheckpoint (best val_accuracy), EarlyStopping (patience 2) |

## Requirements

- Python 3.9+
- tensorflow
- numpy
- matplotlib
- scikit-learn

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

1. Run the script (internet is needed on the first run to download the dataset):
```bash
   python sentiment_lstm.py
```
2. If `sentiment_lstm.keras` does not exist, the script trains the model, prints the evaluation results and saves the model and `training_history.png`.
3. When the prompt appears, type a movie review:
```
   Enter review: This movie was absolutely fantastic, I loved every minute

   >>> Sentiment: Positive (94.3% confidence)
```
4. Type `exit` to quit.

To retrain, delete `sentiment_lstm.keras` and run the script again.

## Results

| Metric | Value |
|---|---|
| Training accuracy | XX% |
| Validation accuracy | XX% |
| Test accuracy | XX% |

![Training History](training_history.png)

## Sample Predictions

| Review | Prediction |
|---|---|
| Great acting, a brilliant story and a perfect ending | Positive |
| Terrible plot and awful acting, a complete waste of time | Negative |

## Limitations

- Works on English text only
- Trained on movie reviews, so accuracy may drop on other topics (products, food, tweets)
- Sarcasm and mixed opinions can be misclassified
- Very short inputs (one or two words) give less reliable confidence

## Future Improvements

- Use pretrained embeddings (GloVe / Word2Vec)
- Try GRU or Transformer-based models
- Add a neutral class for mixed reviews
- Deploy as a web app using Flask or Streamlit

## License

This project is for educational purposes.
