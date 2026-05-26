"""
Emotion Classification using Bidirectional LSTM
================================================
Classifies text into one of 6 emotions:
    anger, joy, sadness, fear, love, surprise
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# ── Hyperparameters (override via CLI args) ───────────────────────────────────
DEFAULTS = {
    "max_sequence_length": 100,
    "embedding_dim": 100,
    "lstm_units": 64,
    "dense_units": 32,
    "dropout_rate": 0.3,
    "epochs": 10,
    "batch_size": 32,
    "validation_split": 0.1,
    "early_stopping_patience": 3,
}

LABEL_TO_ID = {"anger": 0, "joy": 1, "sadness": 2, "fear": 3, "love": 4, "surprise": 5}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_train_data(train_path: str):
    """Load training data from a semicolon-delimited .txt file (text;label)."""
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = [line.strip().split(";") for line in lines if line.strip()]
    speeches, labels = zip(*data)

    # Validate labels
    unknown = set(labels) - set(LABEL_TO_ID.keys())
    if unknown:
        raise ValueError(f"Unknown labels found in training data: {unknown}")

    numeric_labels = [LABEL_TO_ID[label] for label in labels]
    return list(speeches), numeric_labels


def load_test_data(test_path: str):
    """Load test data from a CSV file with a 'Speech' column."""
    df = pd.read_csv(test_path)
    if "Speech" not in df.columns:
        raise ValueError("Test CSV must contain a 'Speech' column.")
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────
def build_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


def encode(tokenizer, texts, maxlen):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(vocab_size, cfg):
    model = Sequential([
        Embedding(vocab_size, cfg["embedding_dim"],
                  input_length=cfg["max_sequence_length"]),
        Bidirectional(LSTM(cfg["lstm_units"])),
        Dropout(cfg["dropout_rate"]),
        Dense(cfg["dense_units"], activation="relu"),
        Dropout(cfg["dropout_rate"]),
        Dense(len(LABEL_TO_ID), activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(cfg, train_path, test_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading training data...")
    train_texts, train_labels_numeric = load_train_data(train_path)

    print("Loading test data...")
    test_df = load_test_data(test_path)
    test_texts = test_df["Speech"].tolist()

    # Tokenize
    print("Tokenizing...")
    tokenizer = build_tokenizer(train_texts)
    X_train = encode(tokenizer, train_texts, cfg["max_sequence_length"])
    X_test  = encode(tokenizer, test_texts,  cfg["max_sequence_length"])

    # Labels → one-hot
    y_train = tf.keras.utils.to_categorical(
        train_labels_numeric, num_classes=len(LABEL_TO_ID)
    )

    # Build model
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    model = build_model(vocab_size, cfg)
    model.summary()

    # Callbacks — val_loss now available because validation_split > 0
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=cfg["validation_split"],
        callbacks=callbacks,
    )

    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved → {tokenizer_path}")

    # Predict on test set
    print("\nPredicting on test set...")
    preds = model.predict(X_test)
    predicted_labels = [ID_TO_LABEL[np.argmax(p)] for p in preds]

    test_df["Predicted Emotion"] = predicted_labels
    out_csv = os.path.join(output_dir, "predictions.csv")
    test_df.to_csv(out_csv, index=False)
    print(f"Predictions saved → {out_csv}")
    print("\nPredicted emotions:", predicted_labels)

    return history, model


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Emotion LSTM")
    parser.add_argument("--train",       required=True,  help="Path to train.txt")
    parser.add_argument("--test",        required=True,  help="Path to Data.csv")
    parser.add_argument("--output",      default="models", help="Output directory")
    parser.add_argument("--epochs",      type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lstm_units",  type=int,   default=DEFAULTS["lstm_units"])
    parser.add_argument("--dropout",     type=float, default=DEFAULTS["dropout_rate"])
    parser.add_argument("--max_len",     type=int,   default=DEFAULTS["max_sequence_length"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {**DEFAULTS,
           "epochs": args.epochs,
           "batch_size": args.batch_size,
           "lstm_units": args.lstm_units,
           "dropout_rate": args.dropout,
           "max_sequence_length": args.max_len}
    train(cfg, args.train, args.test, args.output)
