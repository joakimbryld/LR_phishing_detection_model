import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

import helpers as h

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    brier_score_loss,
)


SAVE_DIR = Path("../figures/ablation_outputs")
SAVE_FILE = "ablation_results.csv"
SEED = 42


def make_model_pipeline(cfg, add_struct=True):

    tfidf = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=tuple(cfg["tfidf__ngram_range"]),
        min_df=int(cfg["tfidf__min_df"]),
        max_df=float(cfg["tfidf__max_df"]),
        max_features=int(cfg["tfidf__max_features"]),
        sublinear_tf=bool(cfg["tfidf__sublinear_tf"]),
        norm=str(cfg["tfidf__norm"]),
    )


    feature_parts = [("tfidf", tfidf)]

    # sturctural features
    if add_struct:
        feature_parts.append(("struct_feats", h.StructFeatureExtractor()))

    full_features = FeatureUnion(feature_parts)

    clf = LogisticRegression(
        solver=cfg["lr__solver"],
        penalty=cfg["lr__penalty"],
        C=cfg["lr__C"],
        max_iter=cfg["lr__max_iter"],
        class_weight=cfg["lr__class_weight"],
        random_state=SEED,
    )

    return Pipeline([
        ("features", full_features),
        ("logreg", clf)
    ])


def scan_thresholds(y_actual, probas, step=0.01):
    # scan over threshold
    best_score = -1
    best_threshold = 0.5 

    for thresh in np.arange(0.0, 1.000001, step):
        preds = (probas >= thresh).astype(int)
        _, _, f1s, _ = precision_recall_fscore_support(
            y_actual, preds, average=None, labels=[0, 1], zero_division=0
        )
        if f1s[1] > best_score:
            best_score = f1s[1]
            best_threshold = thresh

    return best_threshold


def collect_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp) 

    prec, rec, f1, recall = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    return {
        "test_acc": acc,
        "test_precision": prec[1],
        "test_recall": rec[1],
        "test_f1": f1[1],
        "test_fpr": fpr,
        "test_fnr": fnr,
        "test_tn": tn,
        "test_fp": fp,
        "test_fn": fn,
        "test_tp": tp,
    }


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges, right=True)

    ece_total = 0.0
    total_count = len(y_true)

    for i in range(1, n_bins + 1):
        in_bin = bin_ids == i
        if np.any(in_bin):
            avg_conf = np.mean(y_prob[in_bin])
            avg_acc = np.mean(y_true[in_bin])
            weight = np.sum(in_bin) / total_count
            ece_total += weight * abs(avg_conf - avg_acc)

    return ece_total


def get_memory_usage(predict_fn, texts):
    texts = np.asarray(texts, dtype=object)
    if len(texts) == 0:
        return 0.0

    not_used = predict_fn(texts[:min(10, len(texts))])  
    tracemalloc.start()
    not_used = predict_fn(texts)
    not_used, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / (1024.0 * 1024.0)  # converting to MB


def run_config(
    run_name,
    add_struct,
    calibrate,
    cfg,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    X_extra=None,
    y_extra=None,
):
    model = make_model_pipeline(cfg, add_struct)
    model.fit(X_train, y_train)

    final_model = model
    if calibrate:
        calibrated_model = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
        calibrated_model.fit(X_val, y_val)
        final_model = calibrated_model

    val_probs = final_model.predict_proba(X_val)[:, 1]
    threshold = scan_thresholds(y_val, val_probs)

    test_probs = final_model.predict_proba(X_test)[:, 1]
    y_pred = (test_probs >= threshold).astype(int)

    def pred_fn(texts):
        return final_model.predict_proba(np.asarray(texts, dtype=object))[:, 1]

    result_row = {
        "name": run_name,
        "calibration": calibrate,
        "val_threshold": threshold,
    }

    result_row.update(collect_metrics(y_test, y_pred))
    result_row["test_auc"] = roc_auc_score(y_test, test_probs)
    result_row["test_ece"] = expected_calibration_error(y_test, test_probs)
    result_row["test_brier"] = brier_score_loss(y_test, test_probs)
    result_row["test_peak_mem_mb"] = get_memory_usage(pred_fn, X_test)


    extra_probs = pred_fn(X_extra)
    extra_preds = (extra_probs >= threshold).astype(int)

    extra_metrics = collect_metrics(y_extra, extra_preds)

    result_row.update({
        "ext_auc": roc_auc_score(y_extra, extra_probs),
        "ext_acc": extra_metrics["test_acc"],
        "ext_precision": extra_metrics["test_precision"],
        "ext_recall": extra_metrics["test_recall"],
        "ext_f1": extra_metrics["test_f1"],
        "ext_fpr": extra_metrics["test_fpr"],
        "ext_fnr": extra_metrics["test_fnr"],
        "ext_tn": extra_metrics["test_tn"],
        "ext_fp": extra_metrics["test_fp"],
        "ext_fn": extra_metrics["test_fn"],
        "ext_tp": extra_metrics["test_tp"],
        "ext_ece": expected_calibration_error(y_extra, extra_probs),
        "ext_brier": brier_score_loss(y_extra, extra_probs),
        "ext_peak_mem_mb": get_memory_usage(pred_fn, X_extra),
    })

    return result_row


def main():
    # laoding best config
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

    # load dataset and split it up
    df_main = h.load_internal_dataset()
    split = h.split_with_overlap_removal(
        df_main, text_col="text", test_size=0.3, val_size=0.5, random_state=SEED
    )

    df = split["df_norm"]
    X = df["text"].to_numpy(dtype=object)
    y = df["label"].to_numpy(dtype=int)

    # train/val/test splits
    X_tr = X[split["train_idx"]]
    y_tr = y[split["train_idx"]]
    X_val = X[split["val_idx_clean"]]
    y_val = y[split["val_idx_clean"]]
    X_test = X[split["test_idx_clean"]]
    y_test = y[split["test_idx_clean"]]

    ext_df = h.load_external_dataset()

    X_ext = ext_df["text"].astype(str).to_numpy()
    y_ext = ext_df["label"].to_numpy()


    experiments = [
        {"name": "Word only + calibration", "include_struct": False, "use_calibration": True},
        {"name": "Word only no calibration", "include_struct": False, "use_calibration": False},
        {"name": "Word+Struct + calibration", "include_struct": True, "use_calibration": True},
        {"name": "Word+Struct no calibration", "include_struct": True, "use_calibration": False},
    ]

    results = []
    for exp in experiments:
        results.append(
            run_config(
                exp["name"],
                exp["include_struct"],
                exp["use_calibration"],
                cfg,
                X_tr, y_tr,
                X_val, y_val,
                X_test, y_test,
                X_ext, y_ext,
            )
        )

    df_results = pd.DataFrame(results)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SAVE_DIR / SAVE_FILE
    df_results.to_csv(output_path, index=False)

    print("Results written to:", output_path)


if __name__ == "__main__":
    main()
