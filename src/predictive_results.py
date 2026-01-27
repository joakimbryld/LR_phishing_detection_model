import numpy as np
from pathlib import Path
import helpers as h
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

THRESHOLD = 0.56


def print_metrics(y_true, prob_pos, label):
    predictions = prob_pos >= THRESHOLD
    acc = accuracy_score(y_true, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, average="binary", pos_label=1, zero_division=0
    )

    print(label)
    print(f"Accuracy: ", acc)
    print(f"Precision:", precision)
    print(f"Recall: ", recall)
    print(f"F1: ", f1)


def main():
    internal_df = h.load_internal_dataset()
    external_df = h.load_external_dataset()

    split = h.split_with_overlap_removal(
        internal_df,
        text_col="text",
        test_size=0.3,
        val_size=0.5,
        random_state=42
    )

    df_normalized = split["df_norm"]
    X = df_normalized["text"].astype(str).values
    y = df_normalized["label"].values

    train_indices = split["train_idx"]
    val_indices = split["val_idx_clean"]
    test_indices = split["test_idx_clean"]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    model = h.build_model()
    model.fit(X_train, y_train)

    final_ready_model = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    final_ready_model.fit(X_val, y_val)

    test_probs = final_ready_model.predict_proba(X_test)[:, 1]
    print_metrics(y_test, test_probs, "Internal dataset")

    X_ext = external_df["text"].astype(str).values
    y_ext = external_df["label"].values
    ext_probs = final_ready_model.predict_proba(X_ext)[:, 1]
    print_metrics(y_ext, ext_probs, "External dataset")


if __name__ == "__main__":
    main()
