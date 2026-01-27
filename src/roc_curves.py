from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helpers as h
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, roc_auc_score

OUT_DIR = Path("../figures/roc_curves")


def plot_roc(y_true, prob_pos, label, ax):
    fpr, tpr, _ = roc_curve(y_true, prob_pos)

    auc = float(roc_auc_score(np.asarray(y_true), np.asarray(prob_pos)))

    ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc:.4f})")


def main():
    df = h.load_internal_dataset()
    df_ext = h.load_external_dataset()
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
    X_ext = df_ext["text"].astype(str).values
    y_ext = df_ext["label"].values
    base_model = h.build_model()
    base_model.fit(X_train, y_train)
    cal = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)
    p_int = cal.predict_proba(X_test)[:, 1]
    p_ext = cal.predict_proba(X_ext)[:, 1]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_roc(y_test, p_int, "Internal", ax)
    plot_roc(y_ext, p_ext, "External", ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=16)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=16)
    ax.legend(loc="lower right", fontsize=14)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "roc_curves_internal_external.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
