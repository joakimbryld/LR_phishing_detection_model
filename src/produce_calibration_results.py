from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helpers as h
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

OUT_DIR = Path("../figures/calibration_results")
THRESHOLD = 0.56


def get_confidence_buckets(y_true, probs, dataset_name, threshold, high_cutoff=0.9, med_cutoff=0.5):
    preds = (probs >= threshold).astype(int)
    
    confidence = np.where(
        preds == 1,
        (probs - threshold) / max(1.0 - threshold, 1e-12),
        (threshold - probs) / max(threshold, 1e-12)
    )
    confidence = np.clip(confidence, 0, 1)
    
    def bucket_label(conf):
        if conf > high_cutoff:
            return f"High (>{high_cutoff:.1f})"
        elif conf >= med_cutoff:
            return f"Medium ({med_cutoff:.1f}-{high_cutoff:.1f})"
        else:
            return f"Low (<{med_cutoff:.1f})"
    
    buckets = np.array([bucket_label(c) for c in confidence])
    
    high_label = f"High (>{high_cutoff:.1f})"
    med_label = f"Medium ({med_cutoff:.1f}-{high_cutoff:.1f})"
    low_label = f"Low (<{med_cutoff:.1f})"
    
    results = []
    for cls, name in [(0, "Legitimate"), (1, "Phishing")]:
        mask = y_true == cls
        n = mask.sum()
        
        if n == 0:
            results.append({
                "dataset": dataset_name, 
                "class": name, 
                "n": 0,
                high_label: 0.0, 
                med_label: 0.0, 
                low_label: 0.0
            })
        else:
            class_buckets = buckets[mask]
            results.append({
                "dataset": dataset_name,
                "class": name,
                "n": int(n),
                high_label: 100 * (class_buckets == high_label).mean(),
                med_label: 100 * (class_buckets == med_label).mean(),
                low_label: 100 * (class_buckets == low_label).mean(),
            })
    
    return pd.DataFrame(results)


def compute_ece(y_true, probs, n_bins=10):
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, bins, right=True)
    
    ece = 0
    for i in range(1, n_bins + 1):
        in_bin = bin_ids == i
        if in_bin.any():
            bin_acc = y_true[in_bin].mean()
            bin_conf = probs[in_bin].mean()
            bin_weight = in_bin.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)
    
    return ece


def plot_calibration_curve(y_true, probs_before, probs_after, out_path, n_bins=10):
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # Before calibration
    frac_pos_before, mean_pred_before = calibration_curve(
        y_true, probs_before, n_bins=n_bins, strategy="uniform"
    )
    sns.lineplot(
        x=mean_pred_before, y=frac_pos_before, 
        marker="s", label="Before calibration", linewidth=2, ax=ax
    )
    
    # After calibration
    frac_pos_after, mean_pred_after = calibration_curve(
        y_true, probs_after, n_bins=n_bins, strategy="uniform"
    )
    sns.lineplot(
        x=mean_pred_after, y=frac_pos_after,
        marker="o", label="After calibration", linewidth=2, ax=ax
    )
    
    ax.set_xlabel("Mean predicted probability", fontsize=20)
    ax.set_ylabel("Fraction of positives", fontsize=20)
    ax.legend(loc="upper left", fontsize=18)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def get_top_tokens(vectorizer, coef, text, n=10):

    vec = vectorizer.transform([text])
    contributions = vec.toarray().ravel() * coef
    
    if not contributions.any():
        return []
    
    top_idx = np.argsort(-contributions)[:n]
    feature_names = vectorizer.get_feature_names_out()
    
    # positive contribs
    tokens = []
    for i in top_idx:
        if contributions[i] > 0:
            tokens.append(feature_names[i])

    return tokens


def collect_examples(df, probs, vectorizer, coef, dataset_name, n_examples=10):

    y = df["label"].values
    texts = df["text"].astype(str).values
    
    top_indices = np.argsort(probs)[-n_examples:][::-1]
    bottom_indices = np.argsort(probs)[:n_examples]
    
    selected = list(top_indices)
    for idx in bottom_indices:
        if idx not in selected:
            selected.append(idx)
    
    examples = []
    for idx in selected:
        text = texts[idx]
        tokens = get_top_tokens(vectorizer, coef, text, n=10)
        # 240 characters
        snippet = text[:240].replace("\n", " ").replace("\r", " ")
        
        examples.append({
            "dataset": dataset_name,
            "true_label": int(y[idx]),
            "p_phish": float(probs[idx]),
            "p_legit": float(1 - probs[idx]),
            "tokens": ", ".join(tokens),
            "snippet": snippet
        })
    
    return examples


def main():
    # load data
    df = h.load_internal_dataset()
    df_ext = h.load_external_dataset()
    
    # split data
    split = h.split_with_overlap_removal(
        df, text_col="text", test_size=0.3, val_size=0.5, random_state=42
    )
    
    df_norm = split["df_norm"]
    X = df_norm["text"].astype(str).values
    y = df_norm["label"].values
    
    X_train = X[split["train_idx"]]
    y_train = y[split["train_idx"]]
    X_val = X[split["val_idx_clean"]]
    y_val = y[split["val_idx_clean"]]
    X_test = X[split["test_idx_clean"]]
    y_test = y[split["test_idx_clean"]]
    

    model = h.build_model()
    model.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    

    probs_before = model.predict_proba(X_test)[:, 1]
    probs_after = calibrated.predict_proba(X_test)[:, 1]
    

    ece_before = compute_ece(y_test, probs_before)
    ece_after = compute_ece(y_test, probs_after)
    brier_before = brier_score_loss(y_test, probs_before)
    brier_after = brier_score_loss(y_test, probs_after)
    
    print("\n[Internal test set]")
    print(f"  ECE before: {ece_before:.5f}")
    print(f"  ECE after:  {ece_after:.5f}")
    print(f"  Brier before: {brier_before:.5f}")
    print(f"  Brier after:  {brier_after:.5f}")
    

    plot_calibration_curve(
        y_test, probs_before, probs_after, 
        OUT_DIR / "reliability_internal.png"
    )
    

    conf_df = get_confidence_buckets(y_test, probs_after, "Internal", THRESHOLD)
    

    df_test = pd.DataFrame({"text": X_test, "label": y_test})
    vectorizer = model.named_steps["features"].transformer_list[0][1]
    coef = model.named_steps["clf"].coef_.ravel()[:len(vectorizer.get_feature_names_out())]
    
    all_examples = collect_examples(df_test, probs_after, vectorizer, coef, "Internal", n_examples=10)
    
    X_ext = df_ext["text"].astype(str).values
    y_ext = df_ext["label"].values
    
    probs_ext_before = model.predict_proba(X_ext)[:, 1]
    probs_ext_after = calibrated.predict_proba(X_ext)[:, 1]
    
    ece_ext_before = compute_ece(y_ext, probs_ext_before)
    ece_ext_after = compute_ece(y_ext, probs_ext_after)
    brier_ext_before = brier_score_loss(y_ext, probs_ext_before)
    brier_ext_after = brier_score_loss(y_ext, probs_ext_after)
    
    print("\n[External dataset]")
    print(f"  ECE before: ", ece_ext_before)
    print(f"  ECE after:  ", ece_ext_after)
    print(f"  Brier before: ", brier_ext_before)
    print(f"  Brier after:  ", brier_ext_after)
    
    plot_calibration_curve(
        y_ext, probs_ext_before, probs_ext_after,
        OUT_DIR / "reliability_external.png"
    )
    
    conf_ext = get_confidence_buckets(y_ext, probs_ext_after, "External", THRESHOLD)
    conf_df = pd.concat([conf_df, conf_ext], ignore_index=True)
    
    df_ext_simple = pd.DataFrame({"text": X_ext, "label": y_ext})
    ext_examples = collect_examples(df_ext_simple, probs_ext_after, vectorizer, coef, "External", n_examples=10)
    all_examples.extend(ext_examples)
    

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conf_df.to_csv(OUT_DIR / "confidence_by_class.csv", index=False)
    pd.DataFrame(all_examples).to_csv(OUT_DIR / "predicted_examples.csv", index=False)
    


if __name__ == "__main__":
    main()
