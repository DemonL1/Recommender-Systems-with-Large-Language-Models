"""
Simple Model Evaluation - No Chinese Characters
Generate English evaluation charts
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

# Set English fonts
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300

from data_loader import MovieLensDataLoader
from model import LightGCN, evaluate_model

def main():
    print("Loading model and data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    loader = MovieLensDataLoader()
    if not loader.load_preprocessed_data():
        print("Failed to load data")
        return
    
    # Load model
    checkpoint = torch.load('best_lightgcn_model.pth', map_location=device)
    config = checkpoint.get('config', {'embedding_dim': 64, 'num_layers': 3, 'dropout': 0.1})
    
    model = LightGCN(
        num_nodes=loader.graph_data.num_nodes,
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_bert=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    
    # Prepare data
    ratings_df = loader.ratings_df
    train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_ratings)}, Test: {len(test_ratings)}")
    
    # Quick evaluation with limited users
    graph_data = loader.graph_data.to(device)
    
    print("Evaluating model...")
    results = {}
    
    for k in [5, 10, 20]:
        print(f"Evaluating K={k}")
        metrics = evaluate_model(
            model, test_ratings, loader.user_id_map, loader.movie_id_map,
            graph_data, id_to_movie=loader.id_to_movie, k=k,
            max_users=200, train_ratings=train_ratings, validation_type='ratings'
        )
        results[k] = metrics
        print(f"K={k}: HR={metrics['hr']:.3f}, NDCG={metrics['ndcg']:.3f}")
    
    # Create visualization
    print("Creating charts...")
    
    # Create output directory
    if not os.path.exists('eval_charts'):
        os.makedirs('eval_charts')
    
    # Performance chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    k_values = [5, 10, 20]
    hr_values = [results[k]['hr'] for k in k_values]
    ndcg_values = [results[k]['ndcg'] for k in k_values]
    recall_values = [results[k]['recall'] for k in k_values]
    precision_values = [results[k]['precision'] for k in k_values]
    
    x = np.arange(len(k_values))
    width = 0.2
    
    ax.bar(x - 1.5*width, hr_values, width, label='Hit Rate@K', alpha=0.8)
    ax.bar(x - 0.5*width, ndcg_values, width, label='NDCG@K', alpha=0.8)
    ax.bar(x + 0.5*width, recall_values, width, label='Recall@K', alpha=0.8)
    ax.bar(x + 1.5*width, precision_values, width, label='Precision@K', alpha=0.8)
    
    ax.set_xlabel('K Value')
    ax.set_ylabel('Performance Score')
    ax.set_title('LightGCN Model Performance Evaluation')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, k in enumerate(k_values):
        ax.text(i - 1.5*width, hr_values[i] + 0.01, f'{hr_values[i]:.3f}', 
               ha='center', va='bottom', fontsize=9)
        ax.text(i - 0.5*width, ndcg_values[i] + 0.01, f'{ndcg_values[i]:.3f}', 
               ha='center', va='bottom', fontsize=9)
        ax.text(i + 0.5*width, recall_values[i] + 0.01, f'{recall_values[i]:.3f}', 
               ha='center', va='bottom', fontsize=9)
        ax.text(i + 1.5*width, precision_values[i] + 0.01, f'{precision_values[i]:.3f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('eval_charts/performance_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = []
    for k in k_values:
        data.append([results[k]['recall'], results[k]['precision'], 
                    results[k]['hr'], results[k]['ndcg']])
    
    sns.heatmap(data, annot=True, fmt='.3f', 
               xticklabels=['Recall@K', 'Precision@K', 'HR@K', 'NDCG@K'],
               yticklabels=[f'K={k}' for k in k_values],
               cmap='YlOrRd', ax=ax)
    ax.set_title('Model Performance Metrics Heatmap')
    
    plt.tight_layout()
    plt.savefig('eval_charts/heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open('eval_charts/results.txt', 'w') as f:
        f.write("LightGCN Model Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        for k in k_values:
            f.write(f"K={k}:\n")
            f.write(f"  Recall@{k}: {results[k]['recall']:.4f}\n")
            f.write(f"  Precision@{k}: {results[k]['precision']:.4f}\n")
            f.write(f"  HR@{k}: {results[k]['hr']:.4f}\n")
            f.write(f"  NDCG@{k}: {results[k]['ndcg']:.4f}\n\n")
    
    print("Evaluation completed!")
    print("Charts saved to eval_charts/")
    print("- performance_chart.png")
    print("- heatmap.png")
    print("- results.txt")

if __name__ == '__main__':
    main()





