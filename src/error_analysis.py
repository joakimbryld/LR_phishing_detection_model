from pathlib import Path

import numpy as np
import pandas as pd
import helpers as h
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

THRESHOLD = 0.56
RANDOM_STATE = 42


def main():
    df = h.load_internal_dataset()
    df["text_norm"] = h.normalize_text_series(df["text"])
    X_all = df["text"].astype(str).values
    y_all = df["label"].values
    text_norm_all = df["text_norm"]

    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.3,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    y_temp = y_all[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    train_norm_set = set(text_norm_all[train_idx])
    val_mask = ~text_norm_all[val_idx].isin(train_norm_set)
    val_idx_clean = val_idx[val_mask.values]
    val_norm_set = set(text_norm_all[val_idx_clean])
    train_val_norm_set = train_norm_set.union(val_norm_set)
    test_mask = ~text_norm_all[test_idx].isin(train_val_norm_set)
    test_idx_clean = test_idx[test_mask.values]

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_val = X_all[val_idx_clean]
    y_val = y_all[val_idx_clean]
    X_test = X_all[test_idx_clean]
    y_test = y_all[test_idx_clean]
    base_model = h.build_model()
    base_model.fit(X_train, y_train)

    calibrator = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)
    prob_pos_test = calibrator.predict_proba(X_test)[:, 1]

    y_pred = (prob_pos_test >= float(THRESHOLD)).astype(int)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    rows = []
    if np.any(fp_mask):
        rows.append(
            pd.DataFrame(
                {
                    "error_type": "FP",
                    "text": X_test[fp_mask],
                    "probability": prob_pos_test[fp_mask],
                }
            )
        )
    if np.any(fn_mask):
        rows.append(
            pd.DataFrame(
                {
                    "error_type": "FN",
                    "text": X_test[fn_mask],
                    "probability": prob_pos_test[fn_mask],
                }
            )
        )

    Path("../figures/error_analysis").mkdir(parents=True, exist_ok=True)
    out_txt = Path("../figures/error_analysis/error_fp_fn_internal.txt")
    if rows:
        df_out = pd.concat(rows, ignore_index=True)
    else:
        df_out = pd.DataFrame(columns=["error_type", "text", "probability"])
    df_out["snippet"] = (
        df_out["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 240)
    )
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Type\tSnippet\tProbability\n")
        for _, row in df_out.iterrows():
            err_type = str(row["error_type"])
            snippet = str(row["snippet"])
            prob = f"{float(row['probability']):.4f}"
            f.write(f"{err_type}\t{snippet}\t{prob}\n")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
