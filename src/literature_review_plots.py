from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = Path("../figures/literature_review_plots")

DL_RESULTS = [
    {"author": "Castillo et al. 2020", "model": "CNN", "accuracy": 91.0, "precision": None, "recall": None, "f1": None},
    {"author": "Castillo et al. 2020", "model": "LSTM", "accuracy": 93.0, "precision": None, "recall": None, "f1": None},
    {"author": "Halgaš et al. 2020", "model": "LSTM", "accuracy": 96.74, "precision": 97.45, "recall": 95.98, "f1": 96.71},
    {"author": "Fang et al. 2018", "model": "BiLSTM", "accuracy": 97.38, "precision": 93.26, "recall": 83.0, "f1": 87.83},
    {"author": "Nguyen et al. 2018", "model": "H-ILSTM", "accuracy": None, "precision": 99.0, "recall": 99.2, "f1": 99.1},
    {"author": "Phomkeona et al. 2020", "model": "MLP", "accuracy": 92.86, "precision": 91.27, "recall": 92.86, "f1": None},
    {"author": "Soon et al. 2020", "model": "RNN", "accuracy": 94.22, "precision": None, "recall": None, "f1": None},
    {"author": "Samarthrao & Rohokale 2022", "model": "CNN+RNN", "accuracy": 92.78, "precision": 93.28, "recall": 92.12, "f1": 92.55},
    {"author": "Dewis & Viana 2022", "model": "LSTM", "accuracy": 99.0, "precision": None, "recall": None, "f1": None},
    {"author": "Korkmaz et al. 2022", "model": "G+C+DNN", "accuracy": 98.37, "precision": None, "recall": None, "f1": None},
    {"author": "Zhu et al. 2023", "model": "CNN+BiLSTM", "accuracy": 99.85, "precision": 99.72, "recall": 99.95, "f1": 99.83},
    {"author": "Nooraee & Ghaffari 2022", "model": "LSTM", "accuracy": 99.42, "precision": 100.0, "recall": 100.0, "f1": 70.0},
    {"author": "Muralidharan & Nissim 2023", "model": "Hybrid ML", "accuracy": None, "precision": None, "recall": 94.04, "f1": 96.8},
    {"author": "Jafar et al. 2022", "model": "GRU", "accuracy": 98.3, "precision": 98.6, "recall": None, "f1": 98.3},
    {"author": "Rafat et al. 2022", "model": "LSTM", "accuracy": None, "precision": 95.26, "recall": 97.18, "f1": 96.0},
    {"author": "Ghaleb et al. 2022", "model": "MLP", "accuracy": 97.5, "precision": None, "recall": None, "f1": 96.0},
    {"author": "Bountakas & Xenakis 2023", "model": "Hybrid", "accuracy": 99.07, "precision": 99.06, "recall": 99.07, "f1": 99.07},
    {"author": "Sahingoz et al. 2024", "model": "CNN", "accuracy": 98.74, "precision": None, "recall": None, "f1": None},
    {"author": "Somesha & Pais 2024", "model": "DNN", "accuracy": None, "precision": 99.97, "recall": 99.38, "f1": 99.68},
    {"author": "Linh et al. 2024", "model": "CNN", "accuracy": 98.4, "precision": 98.4, "recall": 98.4, "f1": None},
    {"author": "Chinta et al. 2025", "model": "BERT-LSTM", "accuracy": 99.55, "precision": 99.61, "recall": 99.55, "f1": 99.24},
    {"author": "Pimpason et al. 2025", "model": "LSTM", "accuracy": 99.92, "precision": 99.85, "recall": 99.50, "f1": 99.67},
    {"author": "Arthy et al. 2025", "model": "Hybrid ML", "accuracy": 99.8, "precision": 99.3, "recall": 98.9, "f1": 99.1},
    {"author": "Kulal et al. 2025", "model": "MLP", "accuracy": 99.77, "precision": None, "recall": None, "f1": 69.03},
    {"author": "Jamal et al. 2024", "model": "BERT family", "accuracy": 99.0, "precision": 99.0, "recall": 99.0, "f1": 98.0},
    {"author": "Meléndez et al. 2024", "model": "BERT family", "accuracy": 99.43, "precision": 99.64, "recall": 99.74, "f1": 99.51},
    {"author": "Abdulraheem et al. 2022", "model": "MLP", "accuracy": None, "precision": 96.80, "recall": 96.80, "f1": None},
    {"author": "Bispo et al. 2025", "model": "BERT family", "accuracy": 99.27, "precision": None, "recall": None, "f1": 99.3},
]

ML_RESULTS = [
    {"author": "Tian et al. (2025)", "model": "Ensemble", "accuracy": 99.79, "precision": 98.82, "recall": 98.76, "f1": 98.87},
    {"author": "Tian et al. (2025)", "model": "NB", "accuracy": 96.10, "precision": 91.63, "recall": 89.72, "f1": 94.31},
    {"author": "Tian et al. (2025)", "model": "XGBoost", "accuracy": 96.42, "precision": 95.21, "recall": 96.73, "f1": 95.74},
    {"author": "Kshirsagar et al. (2025)", "model": "Ensemble", "accuracy": 99.04, "precision": 99.04, "recall": 98.95, "f1": 98.99},
    {"author": "Prosun et al. (2021)", "model": "Ensemble", "accuracy": 98.00, "precision": None, "recall": None, "f1": None},
    {"author": "Sarkar et al. (2023)", "model": "LR", "accuracy": 97.00, "precision": 100.0, "recall": None, "f1": None},
    {"author": "Sarkar et al. (2023)", "model": "KNN", "accuracy": 94.00, "precision": 100.0, "recall": None, "f1": None},
    {"author": "Abdulraheem et al. (2022)", "model": "LMT", "accuracy": None, "precision": 96.90, "recall": 96.90, "f1": None},
    {"author": "Keskin & Sevli (2024)", "model": "SVM", "accuracy": 98.74, "precision": 98.86, "recall": 99.89, "f1": 99.29},
    {"author": "Keskin & Sevli (2024)", "model": "RF", "accuracy": 98.83, "precision": 98.78, "recall": 99.89, "f1": 98.26},
    {"author": "Keskin & Sevli (2024)", "model": "GBT", "accuracy": 94.97, "precision": None, "recall": None, "f1": None},
    {"author": "Keskin & Sevli (2024)", "model": "LR", "accuracy": 97.66, "precision": 97.75, "recall": 99.89, "f1": 98.68},
]

DATASETS = [
    "Enron", "APWG", "Own Dataset",
    "Enron", "APWG", "Own Dataset",
    "SpamAssassin", "Enron", "Nazario",
    "IWSPA-AP 2018", "Enron", "SpamAssassin",
    "IWSPA-AP 2018",
    "Enron", "Kyushu", "Own Dataset",
    "CSDMC2010 SPAM",
    "Kaggle",
    "Kaggle",
    "PhishTank",
    "PhishTank",
    "Kaggle",
    "VirusTotal",
    "ISC2016",
    "SpamAssassin",
    "SpamBase", "SpamAssassin", "UK-2011",
    "Enron", "SpamAssassin",
    "PhishTank", "CommonCrawl",
    "SpamAssassin", "Own Dataset",
    "--",
    "--",
    "Figshare",
    "Millersmile", "Nazario", "Enron",
    "Kaggle",
    "Enron", "Email-trainingdata-20k", "Phishing Monkey", "Kaggle",
    "PhishTank",
    "PhishTank", "SpamAssassin",
    "Kaggle", "Enron",
    "Kaggle", "Enron",
    "Kaggle", "Enron",
    "TREC 2007", "Enron", "External",
    "Own Dataset",
    "Kaggle",
    "Kaggle",
    "PhishTank",
    "Spam Dataset",
    "Spam Dataset",
    "Spam Dataset",
    "Spam Dataset",
]

def build_heatmap(data, labels, metrics, out_path):
    df_long = pd.DataFrame(data)
    heatmap_data = (
        df_long.pivot(index="model", columns="metric", values="value")
        .reindex(index=labels, columns=metrics)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="crest_r",
        vmin=97.0,
        vmax=100.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Score (%)"},
        mask=heatmap_data.isna(),
        annot_kws={"size": 13},
    )
    ax.set_ylabel("Model (paper)", fontsize=18)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=13, rotation=30)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def top5_heatmap(results, out_path):
    df = pd.DataFrame(results)
    for col in ["accuracy", "precision", "recall", "f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = df[col].notna() & (df[col] <= 1.0)
        df.loc[mask, col] = df.loc[mask, col] * 100.0

    rank_score = df["f1"].fillna(df["accuracy"])
    df = df.assign(_rank=rank_score).sort_values("_rank", ascending=False).head(5).drop(columns=["_rank"])

    metric_cols = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
    ]
    rows = []
    labels = []
    for _, row in df.iterrows():
        label = f"{row['model']}\n({row['author']})"
        labels.append(label)
        for col, name in metric_cols:
            v = row[col]
            if pd.notna(v):
                rows.append({"model": label, "metric": name, "value": float(v)})

    build_heatmap(rows, labels, [name for _, name in metric_cols], out_path)


def create_deeplearning_plot():
    top5_heatmap(DL_RESULTS, OUT_DIR / "dl_top5_grouped.png")


def create_ml_plot():
    top5_heatmap(ML_RESULTS, OUT_DIR / "ml_performance_grouped.png")


def run_dataset_plot():
    datasets_clean = [d for d in DATASETS if d != "--"]
    counts = Counter(datasets_clean)

    labels, values = zip(*counts.most_common())
    values = np.array(values)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(x=list(labels), y=values, ax=ax, color=sns.color_palette("deep")[0])

    ax.set_ylabel("Number of dataset occurrences", fontsize=18)
    ax.set_xlabel("Dataset", fontsize=18)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=13)

    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{int(h)}", ha="center", va="bottom", fontsize=14)

    out_path = OUT_DIR / "dataset_bar_chart.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    create_deeplearning_plot()
    create_ml_plot()
    run_dataset_plot()
