# Property Address Classifier

Classifies raw Indian property addresses into `flat`, `houseorplot`, `landparcel`, `commercial unit`, or `others`.

## Stack
TF-IDF (word + char n-grams) + engineered keyword/numeric features → LightGBM, served via Streamlit.

## Setup
\`\`\`bash
pip install -r requirements.txt
python train.py            # trains model, saves best_model/
streamlit run app.py        # launches GUI at localhost:8501
\`\`\`

## Files
| File | Purpose |
|---|---|
| `dataset_prep.py` | Text cleaning, feature engineering, train/val/test split |
| `train.py` | Trains LightGBM, saves artifacts + confusion matrix |
| `app.py` | Streamlit inference UI |
| `approach.txt` | Methodology and results |
| `best_model/` | Saved model, vectorizers, scaler, label encoder |

## Results
Test set (n=1771): **Macro F1 0.87**, Accuracy 0.89. See `approach.txt`.

## Notes
- Low-confidence `others` predictions are corrected via keyword override.
- Random seed fixed (`SEED=42`) for reproducibility.
