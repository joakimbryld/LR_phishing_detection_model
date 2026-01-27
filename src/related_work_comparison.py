import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("../figures/related_work_comparison")
OUTPUT_FILE = 'external_metrics_vs_related.png'


def main():
    related_work_results = {
        'accuracy': 0.634,
        'precision': 0.685,
        'recall': 0.634,
        'f1': 0.607,
        'auc': 0.76,
    }
    proposed_model_results = {
        'accuracy': 0.748,
        'precision': 0.690769,
        'recall': 0.898,
        'f1': 0.78087,
        'auc': 0.89554,
    }

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    rows = []
    for i in range(len(metrics)):
        m = metrics[i]
        rows.append({
            'metric': labels[i],
            'value': float(related_work_results[m]) * 100.0,
            'model': 'Kshirsagar et al. (2025) – Ensemble',
        })
        rows.append({
            'metric': labels[i],
            'value': float(proposed_model_results[m]) * 100.0,
            'model': 'Proposed model – LR',
        })
    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11.5, 6))
    sns.barplot(data=df, x='metric', y='value', hue='model', ax=ax)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel('Performance (%)', fontsize=18)
    ax.set_xlabel('', fontsize=18)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=14)
    legend = ax.legend(loc='upper left', frameon=True, fontsize=14)
    legend.get_frame().set_facecolor('white')


    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.0,
                f'{h:.1f}',
                ha='center',
                va='bottom',
                fontsize=13,
            )


    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / OUTPUT_FILE
    fig.savefig(outpath, dpi=300)
    plt.close(fig)



if __name__ == '__main__':
    main()
