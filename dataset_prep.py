import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)

CATEGORIES = ["flat", "houseorplot", "landparcel", "commercial unit", "others"]
ENG_COLS = ["address_length", "has_pincode", "math_density",
            "is_flat_kw", "is_comm_kw", "is_land_kw", "has_structure"]

_pincode = re.compile(r"\b\d{6}\b")
_math_chars = re.compile(r"[\d/\-]")
_punct = re.compile(r"[^a-z0-9\s]")
_ws = re.compile(r"\s+")
_alpha_digit = re.compile(r"(?<=[a-z])(?=[0-9])")
_digit_alpha = re.compile(r"(?<=[0-9])(?=[a-z])")

_flat_kw = re.compile(r"\b(apt|apts|bhk|chawl|soc|bldg|flr|residency)\b", re.IGNORECASE)
_comm_kw = re.compile(r"\b(gala|ofc|godown|shed|cplx)\b", re.IGNORECASE)
_land_kw = re.compile(r"\b(khasra|patti|kheti|krishi|gat)\b", re.IGNORECASE)
_structure_kw = re.compile(r"\b(house|bungalow|villa|apartment|flat|residency|chawl)\b", re.IGNORECASE)


def clean(text):
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    t = _punct.sub(" ", t)
    t = _alpha_digit.sub(" ", t)
    t = _digit_alpha.sub(" ", t)
    t = _ws.sub(" ", t).strip()
    return t


def eng_features(raw_series):
    lengths = raw_series.str.len()
    return pd.DataFrame({
        "address_length": lengths,
        "has_pincode": raw_series.apply(lambda s: int(bool(_pincode.search(s)))),
        "math_density": raw_series.apply(lambda s: len(_math_chars.findall(s)) / max(len(s), 1)),
        "is_flat_kw": raw_series.apply(lambda s: int(bool(_flat_kw.search(s)))),
        "is_comm_kw": raw_series.apply(lambda s: int(bool(_comm_kw.search(s)))),
        "is_land_kw": raw_series.apply(lambda s: int(bool(_land_kw.search(s)))),
        "has_structure": raw_series.apply(lambda s: int(bool(_structure_kw.search(s)))),
    })


def load_data(csv_path, text_col="property_address", label_col="categories"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].str.strip() != ""].reset_index(drop=True)
    df["text"] = df[text_col].apply(clean)

    le = LabelEncoder()
    le.fit(CATEGORIES)
    df["label"] = le.transform(df[label_col])

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

    for split in (train, val, test):
        split[ENG_COLS] = eng_features(split[text_col])

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        le,
    )
