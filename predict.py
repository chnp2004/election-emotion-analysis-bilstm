"""
predict.py — Run emotion prediction on new text using a saved model.

Usage:
    python predict.py --model models/best_model.keras \
                      --tokenizer models/tokenizer.pkl \
                      --text "I am so happy today!"

    python predict.py --model models/best_model.keras \
                      --tokenizer models/tokenizer.pkl \
                      --csv path/to/speeches.csv \
                      --col Speech
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

LABEL_TO_ID = {"anger": 0, "joy": 1, "sadness": 2, "fear": 3, "love": 4, "surprise": 5}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def load_artifacts(model_path, tokenizer_path):
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict(texts, model, tokenizer, max_len=100):
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len)
    preds = model.predict(padded)
    labels = [ID_TO_LABEL[np.argmax(p)] for p in preds]
    confidences = [float(np.max(p)) for p in preds]
    return labels, confidences


def main():
    parser = argparse.ArgumentParser(description="Emotion Prediction")
    parser.add_argument("--model",     required=True, help="Path to saved .keras model")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.pkl")
    parser.add_argument("--max_len",   type=int, default=100)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single text string to classify")
    group.add_argument("--csv",  help="CSV file with texts to classify")
    parser.add_argument("--col", default="Speech", help="Column name in CSV (default: Speech)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    model, tokenizer = load_artifacts(args.model, args.tokenizer)

    if args.text:
        labels, confs = predict([args.text], model, tokenizer, args.max_len)
        print(f"Emotion : {labels[0]}")
        print(f"Confidence: {confs[0]:.2%}")
    else:
        df = pd.read_csv(args.csv)
        texts = df[args.col].tolist()
        labels, confs = predict(texts, model, tokenizer, args.max_len)
        df["Predicted Emotion"] = labels
        df["Confidence"] = confs
        df.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")
        print(df[["Speech", "Predicted Emotion", "Confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
