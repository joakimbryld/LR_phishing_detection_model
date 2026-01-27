from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helpers as h
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

OUT_DIR = Path("../figures/sampling_strategies_histograms_and_results")
RANDOM_STATE = 42


def plot_score_histogram(probs, y_true, threshold, out_path, bins=40):
    y_values = np.asarray(y_true)
    prob_values = np.asarray(probs)

    phish_mask = y_values == 1
    ham_mask = y_values == 0

    plt.figure(figsize=(8, 5))

    ham_probs = prob_values[ham_mask]
    phish_probs = prob_values[phish_mask]

    sns.histplot(
        ham_probs,
        bins=bins,
        stat="density",
        alpha=0.6,
        label="Legitimate",
        kde=False,
    )
    sns.histplot(
        phish_probs,
        bins=bins,
        stat="density",
        alpha=0.6,
        label="Phish",
        kde=False,
    )

    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    plt.xlabel("Predicted probability of phishing", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.legend(loc="upper left", fontsize=13)
    plt.tight_layout()

    parent_dir = out_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(out_path), dpi=300)
    plt.close()


def compute_metrics(y_true, y_pred):
    correct_mask = y_true == y_pred
    accuracy = float(np.mean(correct_mask))

    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    tp = conf_matrix[1, 1]

    false_positive_rate = fp / (fp + tn)



    false_negative_rate = fn / (fn + tp)

    precision, recall, f1, not_used = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=[0, 1],
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision_phish": precision[1],
        "recall_phish": recall[1],
        "f1_phish": f1[1],
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
    }

    return metrics


def find_best_threshold(y_val, val_probs, step=0.01):
    thresholds = np.arange(0.0, 1.000001, step)

    best_f1_score = -1.0
    best_threshold = 0.5

    for current_threshold in thresholds:
        predictions = val_probs >= current_threshold
        predictions = predictions.astype(int)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_val,
            predictions,
            average=None,
            labels=[0, 1],
            zero_division=0,
        )

        f1_phish = float(f1[1])

        if f1_phish > best_f1_score:
            best_f1_score = f1_phish
            best_threshold = float(current_threshold)

    return best_threshold


def run_strategy(
    name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    df_ext,
    out_dir,
):
    model = h.build_model()
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)
    val_phish_probs = val_probabilities[:, 1]

    threshold = find_best_threshold(y_val, val_phish_probs)

    test_probabilities = model.predict_proba(X_test)
    test_phish_probs = test_probabilities[:, 1]

    test_predictions = test_phish_probs >= threshold
    test_predictions = test_predictions.astype(int)

    test_metrics = compute_metrics(y_test, test_predictions)

    safe_name = name.replace(" ", "_")
    test_plot_path = out_dir / name / f"{safe_name}_test_hist_thr_{threshold:.2f}.png"

    plot_score_histogram(
        probs=test_phish_probs,
        y_true=y_test,
        threshold=threshold,
        out_path=test_plot_path,
    )

    result_row = {
        "strategy": name,
        "val_thr": threshold,
        "test_acc": test_metrics["accuracy"],
        "test_precision": test_metrics["precision_phish"],
        "test_recall": test_metrics["recall_phish"],
        "test_f1": test_metrics["f1_phish"],
        "test_fpr": test_metrics["fpr"],
        "test_fnr": test_metrics["fnr"],
    }

    ext_texts = df_ext["text"].to_numpy()
    ext_labels = df_ext["label"].to_numpy()

    ext_probabilities = model.predict_proba(ext_texts)
    ext_phish_probs = ext_probabilities[:, 1]

    ext_predictions = ext_phish_probs >= threshold
    ext_predictions = ext_predictions.astype(int)

    ext_metrics = compute_metrics(ext_labels, ext_predictions)

    ext_plot_path = out_dir / name / f"{safe_name}_external_hist_thr_{threshold:.2f}.png"

    plot_score_histogram(
        probs=ext_phish_probs,
        y_true=ext_labels,
        threshold=threshold,
        out_path=ext_plot_path,
    )

    result_row["ext_thr"] = threshold
    result_row["ext_acc"] = ext_metrics["accuracy"]
    result_row["ext_precision"] = ext_metrics["precision_phish"]
    result_row["ext_recall"] = ext_metrics["recall_phish"]
    result_row["ext_f1"] = ext_metrics["f1_phish"]
    result_row["ext_fpr"] = ext_metrics["fpr"]
    result_row["ext_fnr"] = ext_metrics["fnr"]

    return result_row


def main():
    internal_df = h.load_internal_dataset()
    external_df = h.load_external_dataset()
    split_data = h.split_with_overlap_removal(
        internal_df,
        text_col="text",
        test_size=0.3,
        val_size=0.5,
        random_state=RANDOM_STATE,
    )

    normalized_df = split_data["df_norm"]

    all_texts = normalized_df["text"].values
    all_labels = normalized_df["label"].values

    train_indices = split_data["train_idx"]
    val_indices = split_data["val_idx_clean"]
    test_indices = split_data["test_idx_clean"]

    X_train = all_texts[train_indices]
    y_train = all_labels[train_indices]

    X_val = all_texts[val_indices]
    y_val = all_labels[val_indices]

    X_test = all_texts[test_indices]
    y_test = all_labels[test_indices]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    size_info = []

    imbalanced_row = run_strategy(
        "Imbalanced",
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        external_df,
        OUT_DIR,
    )
    results.append(imbalanced_row)

    imbalanced_sizes = {
        "strategy": "Imbalanced",
        "ham": int((y_train == 0).sum()),
        "phish": int((y_train == 1).sum()),
        "total": int(len(y_train)),
    }
    size_info.append(imbalanced_sizes)

    phish_indices = np.where(y_train == 1)[0]
    ham_indices = np.where(y_train == 0)[0]

    if len(phish_indices) > 0:
        if len(ham_indices) > 0:
            rng = np.random.RandomState(RANDOM_STATE)

            if len(ham_indices) > len(phish_indices):
                selected_ham = rng.choice(ham_indices, size=len(phish_indices), replace=False)
                balanced_indices = np.concatenate([phish_indices, selected_ham])
            else:
                selected_phish = rng.choice(phish_indices, size=len(ham_indices), replace=False)
                balanced_indices = np.concatenate([ham_indices, selected_phish])

            rng.shuffle(balanced_indices)

            X_under = X_train[balanced_indices]
            y_under = y_train[balanced_indices]

            under_row = run_strategy(
                "RandomUndersampling",
                X_under,
                y_under,
                X_val,
                y_val,
                X_test,
                y_test,
                external_df,
                OUT_DIR,
            )
            results.append(under_row)

            under_sizes = {
                "strategy": "RandomUndersampling",
                "ham": int((y_under == 0).sum()),
                "phish": int((y_under == 1).sum()),
                "total": int(len(y_under)),
            }
            size_info.append(under_sizes)

            # oversampling
            if len(ham_indices) > len(phish_indices):
                sampled_phish = rng.choice(phish_indices, size=len(ham_indices), replace=True)
                oversampled_indices = np.concatenate([ham_indices, sampled_phish])
            else:
                sampled_ham = rng.choice(ham_indices, size=len(phish_indices), replace=True)
                oversampled_indices = np.concatenate([phish_indices, sampled_ham])

            rng.shuffle(oversampled_indices)

            X_over = X_train[oversampled_indices]
            y_over = y_train[oversampled_indices]

            over_row = run_strategy(
                "RandomOversampling",
                X_over,
                y_over,
                X_val,
                y_val,
                X_test,
                y_test,
                external_df,
                OUT_DIR,
            )
            results.append(over_row)

            over_sizes = {
                "strategy": "RandomOversampling",
                "ham": int((y_over == 0).sum()),
                "phish": int((y_over == 1).sum()),
                "total": int(len(y_over)),
            }
            size_info.append(over_sizes)

    results_df = pd.DataFrame(results)
    results_path = OUT_DIR / "strategy_performance.csv"
    results_df.to_csv(results_path, index=False)


    sizes_df = pd.DataFrame(size_info)

    
    melted = sizes_df.melt(
        id_vars=["strategy"],
        value_vars=["ham", "phish"],
        var_name="class",
        value_name="count",
    )

    melted["strategy"] = melted["strategy"].replace({
        "RandomUndersampling": "Random undersampling",
        "RandomOversampling": "Random oversampling",
    })

    melted["class"] = melted["class"].replace({
        "ham": "Legitimate",
        "phish": "Phish",
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="strategy", y="count", hue="class", ax=ax)

    ax.set_xlabel("Sampling strategy", fontsize=16)
    ax.set_ylabel("Number of samples", fontsize=16)

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.tight_layout()
    size_plot_path = OUT_DIR / "sampling_strategy_sizes.png"
    plt.savefig(size_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {size_plot_path}")


if __name__ == "__main__":
    main()
