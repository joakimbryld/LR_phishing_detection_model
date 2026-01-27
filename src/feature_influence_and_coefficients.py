from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helpers as h

OUT_DIR = Path("../figures/feature_influence")
TOP_N = 20


def plot_barh(labels, values, xlabel, out_path):
    values = np.asarray(values, dtype=float)
    labels = list(labels)
    labels = labels[::-1]
    values = values[::-1]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x=values, y=labels, orient="h")
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel("")
    plt.yticks(fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_feature_influence_plots(
    top_phish_tokens,
    top_phish_coefs,
    top_legit_tokens,
    top_legit_coefs,
    struct_names,
    struct_coefs,
    out_dir=OUT_DIR,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phish_path = out_dir / "feature_influence_tokens_phishing.png"
    legit_path = out_dir / "feature_influence_tokens_legitimate.png"
    struct_path = out_dir / "feature_influence_structural.png"
    plot_barh(labels=list(top_phish_tokens), values=np.asarray(top_phish_coefs), xlabel="LR coefficient (log-odds)", out_path=phish_path)
    plot_barh(labels=list(top_legit_tokens), values=np.asarray(top_legit_coefs), xlabel="LR coefficient (log-odds)", out_path=legit_path)
    plot_barh(labels=list(struct_names), values=np.asarray(struct_coefs), xlabel="LR coefficient (log-odds)", out_path=struct_path)
    print(f"Saved: ", phish_path)
    print(f"Saved: ", legit_path)
    print(f"Saved: ", struct_path)


def main():
    df = h.load_internal_dataset()
    X_all = df["text"].astype(str).values
    y_all = df["label"].values
    model = h.build_model()
    model.fit(X_all, y_all)
    word_vec = model.named_steps["features"].transformer_list[0][1]
    clf = model.named_steps["clf"]
    feature_names = word_vec.get_feature_names_out()
    coef = clf.coef_.ravel()
    n_word = len(feature_names)
    struct_names = ["url_count", "digit_ratio", "email_length"]
    word_coefs = coef[:n_word]
    struct_coefs = coef[n_word : n_word + 3]
    top_n = int(TOP_N)
    top_phish_idx = np.argsort(-word_coefs)[:top_n]
    top_legit_idx = np.argsort(word_coefs)[:top_n]
    top_phish_tokens = []
    for idx in top_phish_idx:
        top_phish_tokens.append(feature_names[idx])

    top_legit_tokens = []
    for idx in top_legit_idx:
        top_legit_tokens.append(feature_names[idx])

    top_phish_coefs = word_coefs[top_phish_idx]
    top_legit_coefs = word_coefs[top_legit_idx]
    save_feature_influence_plots(
        top_phish_tokens=top_phish_tokens,
        top_phish_coefs=top_phish_coefs,
        top_legit_tokens=top_legit_tokens,
        top_legit_coefs=top_legit_coefs,
        struct_names=struct_names,
        struct_coefs=struct_coefs,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
