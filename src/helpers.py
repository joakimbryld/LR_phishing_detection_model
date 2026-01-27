from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

FIG_DIR = Path("../figures")
DATA_DIR = Path("../data")


class StructFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import re
        X = np.asarray(X)
        feats = []
        for text in X:
            t = str(text)
            url_count = len(re.findall('http[s]?://', t))
            digit_ratio = sum((ch.isdigit() for ch in t)) / max(len(t), 1)
            length = len(t)
            feats.append([url_count, digit_ratio, length])
        return np.asarray(feats, dtype=float)

def build_model():
    word = TfidfVectorizer(
        analyzer='word',
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=10000,
        sublinear_tf=False,
        norm='l2',
    )
    feats = FeatureUnion([('word', word), ('struct', StructFeatureExtractor())])
    clf = LogisticRegression(
        solver='liblinear',
        penalty='l2',
        C=2.0,
        max_iter=50,
        class_weight='balanced',
        random_state=42,
    )
    return Pipeline([('features', feats), ('clf', clf)])

#phishing.csv has some trouble reading, and is the only one using the exception
def read_csv_safely(path: Path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:

        return pd.read_csv(path, engine='python', on_bad_lines='skip', **{k: v for (k, v) in kwargs.items() if k != 'engine'})

def normalize_text_series(s: pd.Series):
    return s.astype(str).str.lower().str.replace('\\s+', ' ', regex=True).str.strip()

def load_internal_dataset():
    dfs = []
    #load Nazario
    naz_path = Path("../data/Nazario.csv")
    df = read_csv_safely(naz_path).fillna('')
    df['text'] = df['subject'].astype(str) + ' ' + df['body'].astype(str)
    df['label'] = 1
    cols = ['text', 'label']
    dfs.append(df[cols])
    
    #load spamassassin
    sa_path = Path("../data/SpamAssassin.csv")
    df = read_csv_safely(sa_path).fillna('')
    df['text'] = df['subject'].astype(str) + ' ' + df['body'].astype(str)
    df['label'] = 0
    cols = ['text', 'label']
    dfs.append(df[cols])

    #load legit.csv
    legit_path = Path("../data/legit.csv")
    df = read_csv_safely(legit_path).fillna('')
    df['text'] = df['text'].astype(str)
    df['label'] = 0
    cols = ['text', 'label']
    dfs.append(df[cols])

    #load phishing.csv
    phish_path = Path("../data/phishing.csv")
    df = read_csv_safely(phish_path).fillna('')
    df['text'] = df['text'].astype(str)
    df['label'] = 1
    cols = ['text', 'label']
    dfs.append(df[cols])

    df = pd.concat(dfs, ignore_index=True)
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ''].reset_index(drop=True)
    return df

def load_external_dataset():
    # load external set
    path = Path("../data/Phishing_validation_emails.csv")
    df = read_csv_safely(path)
    df = df.copy().fillna('')
    df['text'] = df['Email Text'].astype(str)
    raw = df['Email Type'].astype(str).str.lower().str.strip()
    labels = []
    for x in raw:
        if x == "phishing email":
            labels.append(1)
        elif x == "safe email":
            labels.append(0)
        else:
            labels.append(0)
    df['label'] = labels
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ''].reset_index(drop=True)
    return df[['text', 'label']]

def split_with_overlap_removal(df, text_col='text', test_size=0.3, val_size=0.5, random_state=42):
    df = df.copy()
    df['text_norm'] = normalize_text_series(df[text_col])
    y = df['label'].values
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=random_state)
    y_temp = y[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_size, stratify=y_temp, random_state=random_state)
    train_norm = set(df.loc[train_idx, 'text_norm'])
    val_keep = ~df.loc[val_idx, 'text_norm'].isin(train_norm)
    val_idx_clean = val_idx[val_keep.values]
    val_norm_clean = set(df.loc[val_idx_clean, 'text_norm'])
    test_keep = ~df.loc[test_idx, 'text_norm'].isin(train_norm.union(val_norm_clean))
    test_idx_clean = test_idx[test_keep.values]
    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'val_idx_clean': val_idx_clean,
        'test_idx': test_idx,
        'test_idx_clean': test_idx_clean,
        'df_norm': df,
    }
