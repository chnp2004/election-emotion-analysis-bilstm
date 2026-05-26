# Analyzing the Role of Emotions in Elections using Bi-Directional LSTM

> This codebase is the official implementation of the following IEEE-published paper:
>
> **"Analyzing the Role of Emotions in Elections using Bi-Directional LSTM"**
> Rajini A., Chakravadhanula Naga Pranav, Santi Rohit Rao, Mummadi Sai Prasanna
> Bhavan's Vivekananda College of Science, Humanities and Commerce (OU), Secunderabad, India
> 📄 [View on IEEE Xplore](https://ieeexplore.ieee.org/document/10426511)

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{rajini2024emotions,
  title     = {Analyzing the Role of Emotions in Elections using Bi-Directional LSTM},
  author    = {Rajini, A. and Chakravadhanula, Naga Pranav and Santi, Rohit Rao and Mummadi, Sai Prasanna},
  booktitle = {IEEE Xplore},
  year      = {2024},
  doi       = {10.1109/10426511},
  url       = {https://ieeexplore.ieee.org/document/10426511}
}
```

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

### 2. Train
```bash
python train.py --train data/train.txt --test data/Data.csv --output models/
```

### 3. Predict
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
  --csv       data/speeches.csv       \
  --output    results.csv
```

---

## License

Released for academic and research use. Please cite the original IEEE paper if you build upon this work.
