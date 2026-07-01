import os
import json
import pickle
import warnings
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

from dataset_prep import load_data, CATEGORIES, ENG_COLS

SEED = 42
DATA_CSV = "task_dataset.csv"
MODEL_DIR = "best_model"

np.random.seed(SEED)


def build_features(train, val, test):
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                                sublinear_tf=True, analyzer="word")
    char_vec = TfidfVectorizer(ngram_range=(3, 5), min_df=3, max_features=30000,
                                sublinear_tf=True, analyzer="char_wb")
    scaler = StandardScaler(with_mean=False)

    Xw_tr = word_vec.fit_transform(train["text"])
    Xc_tr = char_vec.fit_transform(train["text"])
    Xe_tr = scaler.fit_transform(train[ENG_COLS].values.astype(float))
    X_tr = sp.hstack([Xw_tr, Xc_tr, sp.csr_matrix(Xe_tr)]).tocsr()

    def transform(df):
        Xw = word_vec.transform(df["text"])
        Xc = char_vec.transform(df["text"])
        Xe = scaler.transform(df[ENG_COLS].values.astype(float))
        return sp.hstack([Xw, Xc, sp.csr_matrix(Xe)]).tocsr()

    return X_tr, transform(val), transform(test), word_vec, char_vec, scaler


def plot_cm(cm, classes, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train, val, test, le = load_data(DATA_CSV)
    print(f"Split — train: {len(train)}  val: {len(val)}  test: {len(test)}")

    X_tr, X_val, X_te, word_vec, char_vec, scaler = build_features(train, val, test)
    y_tr = train["label"].values
    y_val = val["label"].values
    y_te = test["label"].values

    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    sw = np.array([dict(zip(np.unique(y_tr), cw))[y] for y in y_tr])

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(CATEGORIES),
        n_estimators=800, learning_rate=0.04, num_leaves=63,
        subsample=0.85, colsample_bytree=0.8,
        min_child_samples=10, reg_alpha=0.05, reg_lambda=0.1,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_tr, y_tr, sample_weight=sw,
            eval_set=[(X_val, y_val)], eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        y_pred = model.predict(X_te)
        macro_f1 = f1_score(y_te, y_pred, average="macro")

    print(classification_report(y_te, y_pred, target_names=le.classes_, digits=3))
    plot_cm(confusion_matrix(y_te, y_pred), le.classes_, "confusion_matrix.png")
    print(f"Test Macro F1: {macro_f1:.4f}")

    with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/word_vec.pkl", "wb") as f:
        pickle.dump(word_vec, f)
    with open(f"{MODEL_DIR}/char_vec.pkl", "wb") as f:
        pickle.dump(char_vec, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    meta = {"model_type": "lightgbm", "macro_f1": macro_f1, "eng_cols": ENG_COLS}
    with open(f"{MODEL_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Artifacts saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
