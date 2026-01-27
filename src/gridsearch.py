#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import helpers as h
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import FeatureUnion, Pipeline


OUT_DIR = Path("../data/gridsearch_results")
RANDOM_STATE = 42
#grid
TFIDF_MAX_FEATURES = [10000,20000,40000]
TFIDF_MIN_DF = [3,5]
TFIDF_MAX_DF = [0.8,0.9]
TFIDF_NGRAMS = [(1, 1), (1, 2)]
LR_C = [0.5,1,2,4]
LR_CLASS_WEIGHT = [None,"balanced"]
LR_MAX_ITER = [50]
LR_SOLVER = ["liblinear"]


def pick_threshold_by_f1(y_true, probs, step=0.01):
    thresholds = np.arange(0.0, 1.0 + 1e-9, step)
    best_thr = 0.5
    best_f1 = -1.0
    best_fpr = 1.0
    for thr in thresholds:
        y_pred = (probs >= thr).astype(int)
        prec, rec, f1, recall = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
        f1_phish = float(f1[1])
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if f1_phish > best_f1 or (np.isclose(f1_phish, best_f1) and fpr < best_fpr):
            best_f1 = f1_phish
            best_fpr = fpr
            best_thr = float(thr)
    return best_thr, best_f1, best_fpr


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = h.load_internal_dataset()
    split = h.split_with_overlap_removal(df, text_col="text", test_size=0.3, val_size=0.5, random_state=RANDOM_STATE)
    df_n = split["df_norm"]
    train_idx = split["train_idx"]
    val_idx_clean = split["val_idx_clean"]
    test_idx_clean = split["test_idx_clean"]
    X = df_n["text"].to_numpy(dtype=object)
    y = df_n["label"].to_numpy(dtype=int)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx_clean], y[val_idx_clean]
    X_cv = np.concatenate([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])

    max_features = list(TFIDF_MAX_FEATURES)
    min_df = list(TFIDF_MIN_DF)
    max_df = list(TFIDF_MAX_DF)
    ngrams = list(TFIDF_NGRAMS)
    lr_c = list(LR_C)
    max_iter = list(LR_MAX_ITER)
    solvers = list(LR_SOLVER)
    class_weight = list(LR_CLASS_WEIGHT)

    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", lowercase=True)),
        ("struct", h.StructFeatureExtractor()),
    ])
    pipe = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "features__word__ngram_range": ngrams,
        "features__word__min_df": min_df,
        "features__word__max_df": max_df,
        "features__word__max_features": max_features,
        "clf__C": lr_c,
        "clf__class_weight": class_weight,
        "clf__max_iter": max_iter,
        "clf__solver": solvers,
    }
    fold = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
    cv = PredefinedSplit(fold)
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_cv, y_cv)
    best = gs.best_params_
    probs_val = gs.best_estimator_.predict_proba(X_val)[:, 1]
    thr, f1_phish, fpr = pick_threshold_by_f1(y_val, probs_val, step=0.01)
    best_txt = OUT_DIR / "best_config.txt"
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write("Best configuration (sorted by val F1 desc, then val FPR asc)\n")
        f.write("=" * 70 + "\n")
        f.write(f"tfidf__ngram_range: {best.get('features__word__ngram_range')}\n")
        f.write(f"tfidf__min_df: {best.get('features__word__min_df')}\n")
        f.write(f"tfidf__max_df: {best.get('features__word__max_df')}\n")
        f.write(f"tfidf__max_features: {best.get('features__word__max_features')}\n")
        f.write(f"lr__solver: {best.get('clf__solver')}\n")
        f.write(f"lr__C: {best.get('clf__C')}\n")
        f.write(f"lr__class_weight: {best.get('clf__class_weight')}\n")
        f.write(f"lr__max_iter: {best.get('clf__max_iter')}\n")
        f.write("\nValidation (at val-chosen threshold)\n")
        f.write(f"  thr: {thr:.2f}\n")
        f.write(f"  F1(phish): {f1_phish:.4f}\n")
        f.write(f"  FPR:       {fpr:.4f}\n")


if __name__ == "__main__":
    main()
