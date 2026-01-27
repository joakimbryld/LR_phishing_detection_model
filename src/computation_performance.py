
import time
from pathlib import Path
import numpy as np
import helpers as h
from sklearn.calibration import CalibratedClassifierCV

CONFIG_PATH = Path("../data/gridsearch_results/best_config.txt")

def main():
    df = h.load_internal_dataset()
    split = h.split_with_overlap_removal(df, text_col="text", test_size=0.3, val_size=0.5, random_state=42)
    df_n = split["df_norm"]
    train_idx = split["train_idx"]
    val_idx_clean = split["val_idx_clean"]
    test_idx_clean = split["test_idx_clean"]
    X = df_n["text"].astype(str).values
    y = df_n["label"].values
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx_clean]
    y_val = y[val_idx_clean]
    X_test = X[test_idx_clean]
    y_test = y[test_idx_clean]
    cfg = {
        'tfidf__ngram_range': (1, 2),
        'tfidf__min_df': 3,
        'tfidf__max_df': 0.9,
        'tfidf__max_features': 10000,
        'tfidf__sublinear_tf': False,
        'tfidf__norm': 'l2',
        'lr__solver': 'liblinear',
        'lr__penalty': 'l2',
        'lr__C': 2.0,
        'lr__class_weight': 'balanced',
        'lr__max_iter': 50,
    }
    t0 = time.perf_counter()
    base_clf = h.build_model()
    base_clf.fit(X_train, y_train)
    cal_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv="prefit")
    cal_clf.fit(X_val, y_val)
    final_clf = cal_clf
    train_seconds = float(time.perf_counter() - t0)
    t_inf0 = time.perf_counter()
    probs = final_clf.predict_proba(X_test)[:, 1]
    test_seconds = float(time.perf_counter() - t_inf0)
    pred = (probs >= 0.5).astype(int)
    acc = float(np.mean(pred == y_test))
    tes = acc / train_seconds
    ies = acc / test_seconds
    rtde = acc / (test_seconds / len(X_test))
    print("\n[PHILDER efficiency metrics — internal test]")
    print(f"  TES  = ", tes)
    print(f"  IES  = ", ies)
    print(f"  RTDE = ", rtde)


if __name__ == "__main__":
    main()
