"""
LightGCN recommendation system comprehensive evaluation module.
Implements end-to-end performance diagnostics covering recall, precision, ranking, diversity, and visualization. Suitable for research and production benchmarking.

Module design:
- Supports multi-K evaluation, grouped data diagnostics, experimental reproducibility.
- Integrates coverage, diversity, and novelty for holistic system assessment.
- Generates professional plots and detailed reports for publication and system comparison.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import sys
import io
from datetime import datetime
import json
import os

# Set up font and encoding
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from data_loader import MovieLensDataLoader
from model import LightGCN, evaluate_model

class ComprehensiveEvaluator:
    """Comprehensive Performance Evaluator"""
    
    def __init__(self, model_path='best_lightgcn_model.pth', embeddings_path='lightgcn_embeddings.pt'):
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Loaded data and model
        self.loader = None
        self.model = None
        self.embeddings_data = None
        self.train_ratings = None
        self.test_ratings = None
        self.val_recommendations = None
        
        # Evaluation results
        self.evaluation_results = {}
        
    def load_model_and_data(self):
        """Load model and data"""
        print("Loading model and data...")
        print("=" * 60)
        
        # Load data
        print("Loading data...")
        self.loader = MovieLensDataLoader()
        if not self.loader.load_preprocessed_data():
            print("Preprocessed data loading failed")
            return False
        
        # Load model
        print("Loading model...")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            config = checkpoint.get('config', {
                'embedding_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'test_size': 0.2,
                'random_seed': 42
            })
            
            self.model = LightGCN(
                num_nodes=self.loader.graph_data.num_nodes,
                embedding_dim=config['embedding_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                use_bert=False
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded successfully (Training loss: {checkpoint.get('loss', 'N/A')})")
            
        except FileNotFoundError:
            print(f"Model file not found: {self.model_path}")
            return False
        
        # Load embeddings
        print("Loading embeddings...")
        try:
            self.embeddings_data = torch.load(self.embeddings_path, map_location='cpu')
            print("Embeddings loaded successfully")
        except FileNotFoundError:
            print(f"Embedding file not found: {self.embeddings_path}")
            return False
        
        # Prepare datasets
        print("Preparing datasets...")
        ratings_df = self.loader.ratings_df
        recommendations_df = self.loader.recommendations_df
        
        # Split data (consistent with training)
        self.train_ratings, self.test_ratings = train_test_split(
            ratings_df,
            test_size=config.get('test_size', 0.2),
            random_state=config.get('random_seed', 42)
        )
        
        self.val_recommendations = recommendations_df
        
        print(f"  • Training set: {len(self.train_ratings):,} records")
        print(f"  • Test set: {len(self.test_ratings):,} records")
        print(f"  • Validation set: {len(self.val_recommendations):,} records")
        
        return True
    
    def evaluate_all_metrics(self, k_values=[5, 10, 20], max_users=1000):
        """Evaluate all metrics"""
        print(f"\nComprehensive Performance Evaluation (K values: {k_values}, Max users: {max_users})")
        print("=" * 60)
        
        graph_data = self.loader.graph_data.to(self.device)
        
        # 1. Test set evaluation
        print("\nTest Set Evaluation (ratings):")
        test_results = {}
        for k in k_values:
            print(f"\nEvaluating K={k}:")
            metrics = evaluate_model(
                self.model,
                self.test_ratings,
                self.loader.user_id_map,
                self.loader.movie_id_map,
                graph_data,
                id_to_movie=self.loader.id_to_movie,
                k=k,
                max_users=max_users,
                train_ratings=self.train_ratings,
                validation_type='ratings'
            )
            test_results[k] = metrics
        
        # 2. Validation set evaluation
        print(f"\nValidation Set Evaluation (recommendations):")
        val_results = {}
        for k in k_values:
            print(f"\nEvaluating K={k}:")
            metrics = evaluate_model(
                self.model,
                self.val_recommendations,
                self.loader.user_id_map,
                self.loader.movie_id_map,
                graph_data,
                id_to_movie=self.loader.id_to_movie,
                k=k,
                max_users=max_users,
                train_ratings=self.train_ratings,
                validation_type='recommendations'
            )
            val_results[k] = metrics
        
        self.evaluation_results = {
            'test_results': test_results,
            'val_results': val_results,
            'k_values': k_values,
            'max_users': max_users
        }
        
        return self.evaluation_results
    
    def calculate_additional_metrics(self, sample_size=500):
        """Calculate additional evaluation metrics"""
        print(f"\nCalculating Additional Metrics (Sample size: {sample_size})")
        print("-" * 40)
        
        # Randomly sample users
        test_users = list(set(self.test_ratings['userId'].unique()) & 
                         set(self.loader.user_id_map.keys()))
        sample_users = np.random.choice(test_users, min(sample_size, len(test_users)), replace=False)
        
        user_embeddings = self.embeddings_data['user_embeddings']
        movie_embeddings = self.embeddings_data['movie_embeddings']
        
        # Calculate coverage, diversity, and novelty metrics
        coverage_scores = []
        diversity_scores = []
        novelty_scores = []
        
        # Calculate movie popularity (for novelty calculation)
        movie_popularity = self.train_ratings['movieId'].value_counts().to_dict()
        total_interactions = len(self.train_ratings)
        
        print("Calculating coverage, diversity, and novelty...")
        for user_id in tqdm(sample_users):
            if user_id not in self.loader.user_id_map:
                continue
                
            user_idx = self.loader.user_id_map[user_id]
            user_emb = user_embeddings[user_idx]
            
            # Calculate recommendation scores
            scores = torch.matmul(user_emb, movie_embeddings.T)
            top_20_indices = scores.argsort(descending=True)[:20]
            top_20_movies = [self.loader.id_to_movie.get(idx.item()) for idx in top_20_indices]
            top_20_movies = [m for m in top_20_movies if m is not None]
            
            # Coverage: proportion of unique movies in recommendation list
            coverage = len(set(top_20_movies)) / len(top_20_movies) if top_20_movies else 0
            coverage_scores.append(coverage)
            
            # Diversity: average distance between recommended movies
            if len(top_20_movies) > 1:
                movie_indices = [self.loader.movie_id_map.get(m) for m in top_20_movies if m in self.loader.movie_id_map]
                movie_indices = [idx for idx in movie_indices if idx is not None]
                
                if len(movie_indices) > 1:
                    movie_embs = movie_embeddings[movie_indices]
                    # Calculate pairwise cosine similarity
                    similarities = torch.cosine_similarity(movie_embs.unsqueeze(1), movie_embs.unsqueeze(0), dim=2)
                    # Exclude diagonal, calculate average similarity
                    mask = ~torch.eye(len(movie_embs), dtype=bool)
                    avg_similarity = similarities[mask].mean().item()
                    diversity = 1 - avg_similarity  # Diversity = 1 - similarity
                    diversity_scores.append(diversity)
            
            # Novelty: average inverse popularity of recommended movies
            novelty_sum = 0
            valid_movies = 0
            for movie_id in top_20_movies:
                if movie_id in movie_popularity:
                    popularity = movie_popularity[movie_id] / total_interactions
                    novelty_sum += -np.log2(popularity)  # Negative log popularity
                    valid_movies += 1
            
            if valid_movies > 0:
                novelty = novelty_sum / valid_movies
                novelty_scores.append(novelty)
        
        additional_metrics = {
            'coverage': np.mean(coverage_scores) if coverage_scores else 0,
            'diversity': np.mean(diversity_scores) if diversity_scores else 0,
            'novelty': np.mean(novelty_scores) if novelty_scores else 0,
            'sample_size': len(sample_users)
        }
        
        print(f"  • Coverage: {additional_metrics['coverage']:.4f}")
        print(f"  • Diversity: {additional_metrics['diversity']:.4f}")
        print(f"  • Novelty: {additional_metrics['novelty']:.4f}")
        
        self.evaluation_results['additional_metrics'] = additional_metrics
        return additional_metrics
    
    def create_visualizations(self, save_dir='evaluation_plots'):
        """Create visualization plots"""
        print(f"\nGenerating Visualization Plots...")
        print("-" * 40)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Set plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance metrics comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LightGCN Model Performance Evaluation', fontsize=16, fontweight='bold')
        
        k_values = self.evaluation_results['k_values']
        test_results = self.evaluation_results['test_results']
        val_results = self.evaluation_results['val_results']
        
        metrics = ['recall', 'precision', 'hr', 'ndcg']
        metric_names = ['Recall@K', 'Precision@K', 'Hit Rate@K', 'NDCG@K']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            test_values = [test_results[k][metric] for k in k_values]
            val_values = [val_results[k][metric] for k in k_values]
            
            x = np.arange(len(k_values))
            width = 0.35
            
            ax.bar(x - width/2, test_values, width, label='Test Set (ratings)', alpha=0.8)
            ax.bar(x + width/2, val_values, width, label='Validation Set (recommendations)', alpha=0.8)
            
            ax.set_xlabel('K Value')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(k_values)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for j, (test_val, val_val) in enumerate(zip(test_values, val_values)):
                ax.text(j - width/2, test_val + 0.01, f'{test_val:.3f}', 
                       ha='center', va='bottom', fontsize=9)
                ax.text(j + width/2, val_val + 0.01, f'{val_val:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metrics heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Test set heatmap
        test_data = []
        for k in k_values:
            test_data.append([test_results[k][m] for m in metrics])
        
        sns.heatmap(test_data, annot=True, fmt='.3f', 
                   xticklabels=metric_names, yticklabels=[f'K={k}' for k in k_values],
                   cmap='YlOrRd', ax=ax1)
        ax1.set_title('Test Set Performance Heatmap')
        
        # Validation set heatmap
        val_data = []
        for k in k_values:
            val_data.append([val_results[k][m] for m in metrics])
        
        sns.heatmap(val_data, annot=True, fmt='.3f',
                   xticklabels=metric_names, yticklabels=[f'K={k}' for k in k_values],
                   cmap='YlGnBu', ax=ax2)
        ax2.set_title('Validation Set Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Additional metrics radar chart
        if 'additional_metrics' in self.evaluation_results:
            additional = self.evaluation_results['additional_metrics']
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            metrics_radar = ['Coverage', 'Diversity', 'Novelty']
            values = [additional['coverage'], additional['diversity'], 
                     additional['novelty'] / 10]  # Normalize novelty
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
            values += values[:1]  # Close the plot
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label='LightGCN')
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_radar)
            ax.set_ylim(0, 1)
            ax.set_title('Additional Metrics Radar Chart', pad=20)
            ax.grid(True)
            
            plt.savefig(f'{save_dir}/additional_metrics_radar.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {save_dir}/ directory")
    
    def generate_detailed_report(self, save_path='evaluation_report.txt'):
        """Generate detailed evaluation report"""
        print(f"\nGenerating Detailed Evaluation Report...")
        print("-" * 40)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LightGCN Recommendation System - Comprehensive Performance Evaluation Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Evaluation Device: {self.device}")
        report_lines.append("")
        
        # Dataset information
        report_lines.append("Dataset Information")
        report_lines.append("-" * 40)
        report_lines.append(f"Number of users: {len(self.loader.user_id_map):,}")
        report_lines.append(f"Number of movies: {len(self.loader.movie_id_map):,}")
        report_lines.append(f"Training set size: {len(self.train_ratings):,}")
        report_lines.append(f"Test set size: {len(self.test_ratings):,}")
        report_lines.append(f"Validation set size: {len(self.val_recommendations):,}")
        report_lines.append("")
        
        # Model information
        report_lines.append("Model Information")
        report_lines.append("-" * 40)
        report_lines.append(f"Model Type: LightGCN")
        report_lines.append(f"Number of nodes: {self.loader.graph_data.num_nodes:,}")
        report_lines.append(f"Number of edges: {self.loader.graph_data.edge_index.shape[1]:,}")
        report_lines.append(f"Embedding dimension: {self.embeddings_data['user_embeddings'].shape[1]}")
        report_lines.append("")
        
        # Performance metrics
        report_lines.append("Performance Metrics Details")
        report_lines.append("-" * 40)
        
        k_values = self.evaluation_results['k_values']
        test_results = self.evaluation_results['test_results']
        val_results = self.evaluation_results['val_results']
        
        # Test set results
        report_lines.append("Test Set (ratings) Performance:")
        report_lines.append("")
        header = f"{'K':<6} {'Recall@K':<12} {'Precision@K':<15} {'HR@K':<10} {'NDCG@K':<10}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for k in k_values:
            metrics = test_results[k]
            line = f"K={k:<4} {metrics['recall']:.4f}      {metrics['precision']:.4f}          {metrics['hr']:.4f}     {metrics['ndcg']:.4f}"
            report_lines.append(line)
        
        report_lines.append("")
        
        # Validation set results
        report_lines.append("Validation Set (recommendations) Performance:")
        report_lines.append("")
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for k in k_values:
            metrics = val_results[k]
            line = f"K={k:<4} {metrics['recall']:.4f}      {metrics['precision']:.4f}          {metrics['hr']:.4f}     {metrics['ndcg']:.4f}"
            report_lines.append(line)
        
        report_lines.append("")
        
        # Additional metrics
        if 'additional_metrics' in self.evaluation_results:
            additional = self.evaluation_results['additional_metrics']
            report_lines.append("Additional Evaluation Metrics")
            report_lines.append("-" * 40)
            report_lines.append(f"Coverage: {additional['coverage']:.4f}")
            report_lines.append(f"Diversity: {additional['diversity']:.4f}")
            report_lines.append(f"Novelty: {additional['novelty']:.4f}")
            report_lines.append(f"Evaluation sample size: {additional['sample_size']}")
            report_lines.append("")
        
        # Performance analysis
        report_lines.append("Performance Analysis")
        report_lines.append("-" * 40)
        
        # Find best K values
        best_k_recall = max(k_values, key=lambda k: test_results[k]['recall'])
        best_k_ndcg = max(k_values, key=lambda k: test_results[k]['ndcg'])
        best_k_hr = max(k_values, key=lambda k: test_results[k]['hr'])
        
        report_lines.append(f"Best Recall@K: K={best_k_recall} ({test_results[best_k_recall]['recall']:.4f})")
        report_lines.append(f"Best NDCG@K: K={best_k_ndcg} ({test_results[best_k_ndcg]['ndcg']:.4f})")
        report_lines.append(f"Best HR@K: K={best_k_hr} ({test_results[best_k_hr]['hr']:.4f})")
        report_lines.append("")
        
        # Overall assessment
        avg_recall = np.mean([test_results[k]['recall'] for k in k_values])
        avg_ndcg = np.mean([test_results[k]['ndcg'] for k in k_values])
        avg_hr = np.mean([test_results[k]['hr'] for k in k_values])
        
        report_lines.append("Overall Assessment")
        report_lines.append("-" * 40)
        report_lines.append(f"Average Recall: {avg_recall:.4f}")
        report_lines.append(f"Average NDCG: {avg_ndcg:.4f}")
        report_lines.append(f"Average HR: {avg_hr:.4f}")
        report_lines.append("")
        
        if avg_hr > 0.7:
            report_lines.append("Model performance: Excellent - Hit Rate exceeds 70%, high recommendation accuracy")
        elif avg_hr > 0.5:
            report_lines.append("Model performance: Moderate - Hit Rate between 50%-70%, optimization potential exists")
        else:
            report_lines.append("Model performance: Poor - Hit Rate below 50%, improvements needed")
        
        if avg_ndcg > 0.3:
            report_lines.append("Ranking quality: Good - NDCG exceeds 0.3")
        else:
            report_lines.append("Ranking quality: Average - NDCG below 0.3, consider optimizing ranking algorithm")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Detailed report saved to: {save_path}")
        
        # Save JSON results
        json_path = save_path.replace('.txt', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        print(f"JSON results saved to: {json_path}")
        
        return report_lines
    
    def run_comprehensive_evaluation(self, k_values=[5, 10, 20], max_users=1000, 
                                   sample_size=500, save_plots=True):
        """Run full comprehensive evaluation"""
        print("Starting Comprehensive Performance Evaluation")
        print("=" * 80)
        
        # 1. Load model and data
        if not self.load_model_and_data():
            return False
        
        # 2. Evaluate all metrics
        self.evaluate_all_metrics(k_values, max_users)
        
        # 3. Calculate additional metrics
        self.calculate_additional_metrics(sample_size)
        
        # 4. Generate visualizations
        if save_plots:
            self.create_visualizations()
        
        # 5. Generate report
        self.generate_detailed_report()
        
        print("\n" + "=" * 80)
        print("Comprehensive Evaluation Completed!")
        print("=" * 80)
        print("Generated files:")
        print("  • evaluation_report.txt - Detailed text report")
        print("  • evaluation_report.json - JSON format results")
        if save_plots:
            print("  • evaluation_plots/ - Visualization plots directory")
        
        return True

def main():
    """Main function"""
    evaluator = ComprehensiveEvaluator()
    
    # Run comprehensive evaluation
    success = evaluator.run_comprehensive_evaluation(
        k_values=[5, 10, 20],
        max_users=1000,  # Limit number of users for faster evaluation
        sample_size=500,  # Sample size for additional metrics
        save_plots=True
    )
    
    if not success:
        print("Evaluation failed")
        return
    
    # Print key results summary
    print("\nKey Results Summary:")
    print("-" * 40)
    
    results = evaluator.evaluation_results
    test_results = results['test_results']
    
    for k in results['k_values']:
        metrics = test_results[k]
        print(f"K={k}: Recall={metrics['recall']:.3f}, HR={metrics['hr']:.3f}, NDCG={metrics['ndcg']:.3f}")
    
    if 'additional_metrics' in results:
        additional = results['additional_metrics']
        print(f"\nAdditional Metrics: Coverage={additional['coverage']:.3f}, "
              f"Diversity={additional['diversity']:.3f}, Novelty={additional['novelty']:.3f}")

if __name__ == '__main__':
    main()