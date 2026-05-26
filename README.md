# Emotion Classification — Bidirectional LSTM

Classifies text into **6 emotions**: anger · joy · sadness · fear · love · surprise

---

## Project Structure

```
emotion-lstm/
├── train.py                  # CLI training script
├── predict.py                # CLI inference script
├── requirements.txt
├── notebooks/
│   └── emotion_lstm.ipynb    # Interactive notebook version
└── data/                     # (not committed — add locally)
    ├── train.txt             # semicolon-delimited: text;label
    └── Data.csv              # test CSV with a 'Speech' column
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data

**`data/train.txt`** — one sample per line, semicolon-separated:
```
i feel happy today;joy
this makes me so angry;anger
```

**`data/Data.csv`** — must contain a `Speech` column:
```csv
Speech
I really miss my family.
This is the best day ever!
```

### 3. Train
```bash
python train.py --train data/train.txt --test data/Data.csv --output models/
```

Optional overrides:
```bash
python train.py \
  --train data/train.txt \
  --test  data/Data.csv  \
  --epochs 15            \
  --batch_size 64        \
  --dropout 0.4          \
  --lstm_units 128
```

### 4. Predict on new text
```bash
# Single string
python predict.py \
  --model     models/best_model.keras \
  --tokenizer models/tokenizer.pkl    \
  --text "I am so thrilled to be here!"

# CSV batch
python predict.py \
  --model     models/best_model.keras \
  --tokenizer models/tokenizer.pkl    \
  --csv data/new_speeches.csv         \
  --col Speech                        \
  --output results.csv
```

---

## Model Architecture

| Layer | Details |
|---|---|
| Embedding | vocab_size × 100 |
| Bidirectional LSTM | 64 units (×2 directions) |
| Dropout | 0.3 |
| Dense | 32 units, ReLU |
| Dropout | 0.3 |
| Dense (output) | 6 units, Softmax |

---

## Key fixes vs original notebook

| Issue | Fix |
|---|---|
| Hardcoded local Windows paths | Relative paths + CLI arguments |
| EarlyStopping on `val_loss` with no validation data | Added `validation_split=0.1` |
| No model persisted after training | `ModelCheckpoint` saves `best_model.keras` |
| No regularization → potential overfitting | Added `Dropout(0.3)` layers |
| No tokenizer saved → can't run inference later | `tokenizer.pkl` saved via `pickle` |
| No confidence scores | `np.max(softmax_output)` added to output |

---

## Supported Emotions

| Label | ID |
|---|---|
| anger | 0 |
| joy | 1 |
| sadness | 2 |
| fear | 3 |
| love | 4 |
| surprise | 5 |
