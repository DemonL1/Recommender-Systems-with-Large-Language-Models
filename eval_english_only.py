"""
LightGCN Model Evaluation
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set fonts and styling
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Import without Chinese output
import sys
from io import StringIO

class SilentDataLoader:
    """Silent data loader to avoid Chinese output"""
    def __init__(self):
        from data_loader import MovieLensDataLoader
        # Redirect stdout temporarily
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            self.loader = MovieLensDataLoader()
            self.loader.load_preprocessed_data()
        finally:
            sys.stdout = old_stdout
    
    def get_data(self):
        return self.loader

def evaluate_silent(model, test_data, user_map, movie_map, graph_data, k=10, max_users=200):
    """Silent evaluation function"""
    model.eval()
    
    with torch.no_grad():
        # Get embeddings
        all_embeddings = model.get_embeddings(graph_data.edge_index, graph_data.edge_weight)
        user_embeddings = all_embeddings[:len(user_map)]
        movie_embeddings = all_embeddings[len(user_map):]
        
        # Get test users
        test_users = list(set(test_data['userId'].unique()) & set(user_map.keys()))
        if len(test_users) > max_users:
            test_users = test_users[:max_users]
        
        recalls, precisions, hrs, ndcgs = [], [], [], []
        
        for user_id in tqdm(test_users, desc=f"Evaluating K={k}"):
            if user_id not in user_map:
                continue
                
            user_idx = user_map[user_id]
            user_emb = user_embeddings[user_idx]
            
            # Get user test items
            user_test_items = set(test_data[test_data['userId'] == user_id]['movieId'])
            if len(user_test_items) == 0:
                continue
            
            # Calculate scores
            scores = torch.matmul(user_emb, movie_embeddings.T)
            top_k_indices = scores.argsort(descending=True)[:k]
            
            # Get movie IDs
            id_to_movie = {idx: movie_id for movie_id, idx in movie_map.items()}
            top_k_movies = [id_to_movie.get(idx.item()) for idx in top_k_indices]
            top_k_movies = [m for m in top_k_movies if m is not None]
            
            # Calculate metrics
            hits = len(set(top_k_movies) & user_test_items)
            
            recall = hits / len(user_test_items) if len(user_test_items) > 0 else 0
            precision = hits / k if k > 0 else 0
            hr = 1.0 if hits > 0 else 0.0
            
            # NDCG
            dcg = sum(1 / np.log2(i + 2) for i, movie_id in enumerate(top_k_movies) 
                     if movie_id in user_test_items)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(user_test_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            
            recalls.append(recall)
            precisions.append(precision)
            hrs.append(hr)
            ndcgs.append(ndcg)
    
    return {
        'recall': np.mean(recalls) if recalls else 0,
        'precision': np.mean(precisions) if precisions else 0,
        'hr': np.mean(hrs) if hrs else 0,
        'ndcg': np.mean(ndcgs) if ndcgs else 0
    }

def create_charts(results, output_dir='evaluation_results_en'):
    """Create professional English charts"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    k_values = list(results.keys())
    
    # Colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Performance Comparison Chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics = ['recall', 'precision', 'hr', 'ndcg']
    metric_names = ['Recall@K', 'Precision@K', 'Hit Rate@K', 'NDCG@K']
    
    x = np.arange(len(k_values))
    width = 0.2
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[k][metric] for k in k_values]
        bars = ax.bar(x + i*width, values, width, label=name, 
                     color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, v in enumerate(values):
            ax.text(j + i*width, v + 0.005, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('K Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('LightGCN Model Performance Evaluation', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'K={k}' for k in k_values])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', bbox_inches='tight')
    plt.close()
    
    # 2. Metrics Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = []
    for k in k_values:
        data.append([results[k][m] for m in metrics])
    
    sns.heatmap(data, annot=True, fmt='.3f', 
               xticklabels=metric_names, yticklabels=[f'K={k}' for k in k_values],
               cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Performance Score'})
    ax.set_title('Model Performance Metrics Heatmap', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # 3. Performance Radar Chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Use K=10 for radar
    k = 10
    if k in results:
        values = [results[k][m] for m in metrics]
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3, color='#2E86AB')
        ax.fill(angles, values, alpha=0.25, color='#2E86AB')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(f'Model Performance Radar Chart (K={k})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True)
        
        # Add value labels
        for angle, value, name in zip(angles[:-1], values[:-1], metric_names):
            ax.text(angle, value + 0.05, f'{value:.3f}', 
                   ha='center', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_radar.png', bbox_inches='tight')
    plt.close()
    
    # 4. Summary Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('LightGCN Model Evaluation Summary', fontsize=20, fontweight='bold')
    
    # Main performance chart
    ax1 = fig.add_subplot(gs[:, :2])
    
    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        values = [results[k][metric] for k in k_values]
        ax1.plot(k_values, values, 'o-', linewidth=3, markersize=8, 
                label=name, color=color)
        
        # Add value annotations
        for k, v in zip(k_values, values):
            ax1.annotate(f'{v:.3f}', (k, v), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('K Value', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax1.set_title('Performance Trends', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Key metrics cards
    best_k = max(k_values, key=lambda k: results[k]['hr'])
    best_hr = results[best_k]['hr']
    avg_ndcg = np.mean([results[k]['ndcg'] for k in k_values])
    
    metrics_info = [
        ('Best Hit Rate', f'{best_hr:.3f}', f'at K={best_k}'),
        ('Average NDCG', f'{avg_ndcg:.3f}', 'across all K'),
        ('Evaluated Users', '200', 'sample size')
    ]
    
    for i, (title, value, subtitle) in enumerate(metrics_info):
        ax = fig.add_subplot(gs[i, 2])
        ax.text(0.5, 0.6, value, ha='center', va='center', 
               fontsize=24, fontweight='bold', color=colors[i])
        ax.text(0.5, 0.3, title, ha='center', va='center', 
               fontsize=12, fontweight='bold')
        ax.text(0.5, 0.1, subtitle, ha='center', va='center', 
               fontsize=10, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.savefig(f'{output_dir}/summary_dashboard.png', bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved to {output_dir}/:")
    print("- performance_comparison.png")
    print("- metrics_heatmap.png") 
    print("- performance_radar.png")
    print("- summary_dashboard.png")

def save_report(results, output_dir='evaluation_results_en'):
    """Save evaluation report"""
    
    with open(f'{output_dir}/evaluation_report.txt', 'w') as f:
        f.write("LightGCN Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Performance Results:\n")
        f.write("-" * 20 + "\n")
        
        for k in sorted(results.keys()):
            f.write(f"\nK = {k}:\n")
            f.write(f"  Recall@{k}:    {results[k]['recall']:.4f}\n")
            f.write(f"  Precision@{k}: {results[k]['precision']:.4f}\n")
            f.write(f"  Hit Rate@{k}:  {results[k]['hr']:.4f}\n")
            f.write(f"  NDCG@{k}:      {results[k]['ndcg']:.4f}\n")
        
        # Analysis
        best_k = max(results.keys(), key=lambda k: results[k]['hr'])
        f.write(f"\nAnalysis:\n")
        f.write(f"-" * 10 + "\n")
        f.write(f"Best performing K: {best_k} (HR = {results[best_k]['hr']:.4f})\n")
        
        avg_hr = np.mean([results[k]['hr'] for k in results.keys()])
        if avg_hr > 0.7:
            f.write("Overall performance: Excellent (HR > 0.7)\n")
        elif avg_hr > 0.5:
            f.write("Overall performance: Good (HR > 0.5)\n")
        else:
            f.write("Overall performance: Needs improvement (HR < 0.5)\n")
    
    print(f"Report saved to {output_dir}/evaluation_report.txt")

def main():
    print("Starting LightGCN Model Evaluation...")
    print("Loading model and data...")
    
    # Load data silently
    silent_loader = SilentDataLoader()
    loader = silent_loader.get_data()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('best_lightgcn_model.pth', map_location=device)
    config = checkpoint.get('config', {
        'embedding_dim': 64, 'num_layers': 3, 'dropout': 0.1
    })
    
    from model import LightGCN
    model = LightGCN(
        num_nodes=loader.graph_data.num_nodes,
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_bert=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Prepare test data
    ratings_df = loader.ratings_df
    train_ratings, test_ratings = train_test_split(
        ratings_df, test_size=0.2, random_state=42
    )
    
    print(f"Dataset: {len(train_ratings)} train, {len(test_ratings)} test")
    
    # Evaluate for different K values
    graph_data = loader.graph_data.to(device)
    results = {}
    
    print("Running evaluation...")
    for k in [5, 10, 20]:
        print(f"\nEvaluating for K={k}...")
        metrics = evaluate_silent(
            model, test_ratings, loader.user_id_map, loader.movie_id_map,
            graph_data, k=k, max_users=200
        )
        results[k] = metrics
        print(f"Results: HR={metrics['hr']:.3f}, NDCG={metrics['ndcg']:.3f}")
    
    # Create visualizations
    print("\nGenerating charts...")
    create_charts(results)
    
    # Save report
    save_report(results)
    
    print("\nEvaluation completed successfully!")
    print("\nKey Results:")
    for k in sorted(results.keys()):
        print(f"K={k}: HR={results[k]['hr']:.3f}, NDCG={results[k]['ndcg']:.3f}")

if __name__ == '__main__':
    main()










