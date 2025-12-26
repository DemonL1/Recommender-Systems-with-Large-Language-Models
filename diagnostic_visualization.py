"""
Visualization utilities for diagnostic_model_evaluation outputs.

Reads `diagnostic_report.json` and `user_level_metrics.csv`, then produces
publication-ready charts that highlight recall-focused diagnostics:

1. Aggregate Recall Stats: mean / median / p90 per K (clustered bars).
2. Quality Metrics Trend: precision, HR, NDCG vs. K (multi-line).
3. User Recall Distribution: boxplot of per-user recall for each K.
4. Top vs. Worst Users table-style cards saved as an image.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']


def load_diagnostic_data(report_path, user_metrics_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    user_df = pd.read_csv(user_metrics_path)
    return report, user_df


def plot_aggregate_recall(report, output_dir):
    agg = report['aggregate_metrics']
    rows = []
    for k_str, metrics in agg.items():
        k = int(k_str)
        rows.append(
            {
                'K': k,
                'Recall Mean': metrics['recall_mean'],
                'Recall Median': metrics['recall_median'],
                'Recall P90': metrics['recall_p90'],
            }
        )
    df = pd.DataFrame(rows).sort_values('K')
    melted = df.melt(id_vars='K', var_name='Statistic', value_name='Recall')

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=melted, x='K', y='Recall', hue='Statistic', palette='viridis')
    ax.set_title('Recall Distribution Across K', fontsize=16, fontweight='bold')
    ax.set_ylabel('Recall')
    ax.set_xlabel('K')
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center',
            va='bottom',
            fontsize=9,
        )
    ax.legend(title='', loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_recall.png', bbox_inches='tight')
    plt.close()


def plot_quality_trends(report, output_dir):
    agg = report['aggregate_metrics']
    rows = []
    for k_str, metrics in agg.items():
        k = int(k_str)
        rows.append(
            {
                'K': k,
                'Precision': metrics['precision_mean'],
                'Hit Rate': metrics['hr_mean'],
                'NDCG': metrics['ndcg_mean'],
            }
        )
    df = pd.DataFrame(rows).sort_values('K')

    plt.figure(figsize=(8, 5))
    for column, color in zip(['Precision', 'Hit Rate', 'NDCG'], ['#2E86AB', '#A23B72', '#F18F01']):
        plt.plot(df['K'], df[column], 'o-', linewidth=2.5, markersize=5, label=column, color=color)
        for k, v in zip(df['K'], df[column]):
            plt.annotate(f"{v:.3f}", (k, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
    plt.title('Quality Metrics vs. K', fontsize=16, fontweight='bold')
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_trends.png', bbox_inches='tight')
    plt.close()


def plot_user_recall_distribution(user_df, k_values, output_dir):
    recall_cols = [f'recall@{k}' for k in k_values if f'recall@{k}' in user_df.columns]
    if not recall_cols:
        return
    melted = user_df[recall_cols].melt(var_name='Metric', value_name='Recall')
    melted['K'] = melted['Metric'].str.extract(r'@(\d+)').astype(int)

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=melted, x='K', y='Recall', palette='coolwarm')
    sns.stripplot(
        data=melted.sample(min(len(melted), 2000), random_state=42),
        x='K',
        y='Recall',
        color='black',
        alpha=0.15,
        size=2,
    )
    plt.title('Per-User Recall Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('K')
    plt.ylabel('Recall')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'user_recall_distribution.png', bbox_inches='tight')
    plt.close()


def render_user_cards(report, output_dir):
    def format_user(user):
        summary = [
            f"User ID: {user['userId']}",
            f"Test Items: {user['num_test_items']}",
            f"Seen in Train: {user['num_seen_in_train']}",
        ]
        for k in report['k_values']:
            summary.append(
                f"K={k}: hits={user.get(f'hits@{k}', 0)}, "
                f"recall={user.get(f'recall@{k}', 0):.2f}, "
                f"precision={user.get(f'precision@{k}', 0):.2f}"
            )
        return '\n'.join(summary)

    best = report.get('best_users_by_recall', [])[:5]
    worst = report.get('worst_users_by_recall', [])[:5]

    fig, axes = plt.subplots(5, 2, figsize=(12, 14))
    fig.suptitle('Top vs. Bottom Users by Recall', fontsize=18, fontweight='bold')

    for i in range(5):
        for j, users in enumerate([best, worst]):
            ax = axes[i, j]
            ax.axis('off')
            if i < len(users):
                text = format_user(users[i])
                ax.text(
                    0,
                    1,
                    text,
                    ha='left',
                    va='top',
                    fontsize=10,
                    family='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#F7F7F7', edgecolor='#CCCCCC'),
                )
                title = 'Top Performer' if j == 0 else 'Low Performer'
                ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / 'user_recall_cards.png', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create visualizations for diagnostic evaluation.")
    parser.add_argument(
        '--report-path',
        default='evaluation_charts_en/diagnostic/diagnostic_report.json',
        help='Path to diagnostic_report.json',
    )
    parser.add_argument(
        '--user-metrics-path',
        default='evaluation_charts_en/diagnostic/user_level_metrics.csv',
        help='Path to user_level_metrics.csv',
    )
    parser.add_argument(
        '--output-dir',
        default='evaluation_charts_en/diagnostic/visuals',
        help='Directory to store generated charts',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report, user_df = load_diagnostic_data(args.report_path, args.user_metrics_path)
    k_values = report['k_values']

    plot_aggregate_recall(report, output_dir)
    plot_quality_trends(report, output_dir)
    plot_user_recall_distribution(user_df, k_values, output_dir)
    render_user_cards(report, output_dir)

    print(f"Visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()



