from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helpers as h
from sklearn.metrics import log_loss

OUT_DIR = Path("../figures/loss_curve")


def compute_loss_curve(X_tr, y_tr, X_val, y_val, max_iters):
    iterations = []
    train_loss_vals = []
    val_loss_vals = []

    for step in range(1, int(max_iters) + 1):
        iterations.append(step)

        temp_model = h.build_model()
        temp_model.set_params(clf__max_iter=step)
        temp_model.fit(X_tr, y_tr)

        preds_train = temp_model.predict_proba(X_tr)[:, 1]
        preds_val = temp_model.predict_proba(X_val)[:, 1]

        loss_train = log_loss(y_tr, preds_train, labels=[0, 1])
        loss_val = log_loss(y_val, preds_val, labels=[0, 1])

        train_loss_vals.append(loss_train)
        val_loss_vals.append(loss_val)

    return np.array(iterations), np.array(train_loss_vals), np.array(val_loss_vals)


def plot_loss_curve(iters, loss_train, loss_val, save_path):
    sns.set_theme(style="whitegrid")
    data = {
        "iter": iters,
        "train": loss_train,
        "val": loss_val,
    }

    plt.figure(figsize=(9, 6))
    sns.lineplot(x=data["iter"], y=data["train"], linewidth=2, label="Train log-loss")
    sns.lineplot(x=data["iter"], y=data["val"], linewidth=2, label="Validation log-loss")

    plt.xlabel("Training iterations (max_iter)", fontsize=16)
    plt.ylabel("Log-loss", fontsize=16)
    plt.legend(loc="upper right", fontsize=15)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    dataset = h.load_internal_dataset()
    split_data = h.split_with_overlap_removal(dataset, text_col="text", test_size=0.3, val_size=0.5, random_state=42)

    df_prepped = split_data["df_norm"]
    train_ids = split_data["train_idx"]
    val_ids = split_data["val_idx_clean"]

    full_text = df_prepped["text"].astype(str).values
    full_labels = df_prepped["label"].values

    X_train_set = full_text[train_ids]
    y_train_set = full_labels[train_ids]
    X_val_set = full_text[val_ids]
    y_val_set = full_labels[val_ids]

    print(f"  Train size: {len(train_ids)}")
    print(f"  Val size after: {len(val_ids)}")

    steps, train_losses, val_losses = compute_loss_curve(X_train_set, y_train_set, X_val_set, y_val_set, 50)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_output = OUT_DIR / f"loss_curve_nochar_train_val_maxiter_50.png"

    plot_loss_curve(steps, train_losses, val_losses, final_output)

    print(f"Saved: ", final_output)


if __name__ == "__main__":
    main()
