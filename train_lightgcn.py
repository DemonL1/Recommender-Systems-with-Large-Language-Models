"""
MovieLens-LightGCN training and evaluation pipeline.
Implements graph data preprocessing, negative sampling, BPR loss, checkpointing, performance evaluation, and embedding export for research and development.

Highlights:
- Flexible dataset splitting and graph construction for experimental reproducibility and ablation studies.
- Efficient batched negative sampling, fully parameterized BPR loss and checkpoint-saving with early stopping.
- Offline evaluation for both ratings and recommendation log validation.
- Embedding export suitable for downstream analysis and visualization.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import os

from data_loader import MovieLensDataLoader
from model import LightGCN, bpr_loss, create_negative_samples, evaluate_model

class LightGCNDataset(torch.utils.data.Dataset):
    """Standard LightGCN triplet dataset compatible with full-process DataLoader."""
    
    def __init__(self, triplets):
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]

def train_lightgcn(config):
    """Train LightGCN model"""
    print("Starting LightGCN model training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    loader = MovieLensDataLoader()
    
    # Try to load preprocessed data
    if not loader.load_preprocessed_data():
        print("Preprocessed data does not exist, starting data preprocessing...")
        if not loader.load_data():
            print("Data loading failed")
            return None
        
        loader.create_mappings()
        # Pure LightGCN model: do not use BERT, only use ratings and beliefs for training
        loader.use_bert = False  # Disable BERT
        loader.build_graph(use_ratings=True, use_recommendations=False, use_beliefs=True)
        loader.save_preprocessed_data()
    
    # Get graph data
    graph_data = loader.graph_data
    user_id_map = loader.user_id_map
    movie_id_map = loader.movie_id_map
    ratings_df = loader.ratings_df
    recommendations_df = loader.recommendations_df  # For validation set
    
    print(f"Graph data: {graph_data.num_nodes:,} nodes, {graph_data.edge_index.shape[1]:,} edges")
    
    # Move graph data to device
    graph_data = graph_data.to(device)
    
    # Create model (pure LightGCN, no BERT)
    print("Creating pure LightGCN model (without BERT)...")
    model = LightGCN(
        num_nodes=graph_data.num_nodes,
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_bert=False  # Explicitly disable BERT
    ).to(device)
    
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Dataset splitting explanation:
    # - Training set: part of ratings (user actual ratings)
    # - Validation set: recommendations (system recommendation records) - for evaluating recommendation performance
    # - Test set: another part of ratings - for final testing
    
    print("Dataset splitting explanation:")
    print("  • Training set: part of ratings (for model training)")
    print("  • Validation set: recommendations (system recommendation records, for evaluating recommendation performance)")
    print("  • Test set: another part of ratings (for final testing)")
    print()
    
    # Split ratings into training and test sets
    print("Splitting ratings into training and test sets...")
    # Check if there are enough users for stratify (each user needs at least 2 records)
    user_counts = ratings_df['userId'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    if len(valid_users) >= len(ratings_df['userId'].unique()) * 0.5:
        # Most users have multiple records, can use stratify
        ratings_valid = ratings_df[ratings_df['userId'].isin(valid_users)]
        ratings_single = ratings_df[~ratings_df['userId'].isin(valid_users)]
        
        train_valid, test_valid = train_test_split(
            ratings_valid,
            test_size=config['test_size'],
            random_state=config['random_seed'],
            stratify=ratings_valid['userId']
        )
        
        # Data with single records are also split proportionally
        train_single, test_single = train_test_split(
            ratings_single,
            test_size=config['test_size'],
            random_state=config['random_seed']
        )
        
        train_ratings = pd.concat([train_valid, train_single])
        test_ratings = pd.concat([test_valid, test_single])
    else:
        # Most users have only 1 record, do not use stratify
        train_ratings, test_ratings = train_test_split(
            ratings_df,
            test_size=config['test_size'],
            random_state=config['random_seed']
        )
    
    # Use recommendations as validation set
    validation_recommendations = recommendations_df
    
    print(f"Training set (ratings): {len(train_ratings):,} records")
    print(f"Validation set (recommendations): {len(validation_recommendations):,} records")
    print(f"Test set (ratings): {len(test_ratings):,} records")
    
    # Create negative samples
    triplets = create_negative_samples(
        train_ratings, 
        user_id_map, 
        movie_id_map, 
        num_negatives=config['num_negatives']
    )
    
    # Create data loader
    dataset = LightGCNDataset(triplets)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows
    )
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Train one epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in pbar:
            users, pos_items, neg_items = batch
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            user_emb, item_emb = model.encode_minibatch(
                users, 
                torch.cat([pos_items, neg_items]), 
                graph_data.edge_index, 
                graph_data.edge_weight
            )
            
            pos_emb = item_emb[:len(pos_items)]
            neg_emb = item_emb[len(pos_items):]
            
            # Calculate loss
            loss = bpr_loss(user_emb, pos_emb, neg_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        
        # Print training information
        print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, 'best_lightgcn_model.pth')
            print(f"Saved best model (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered (no improvement for {config['patience']} consecutive epochs)")
                break
        
        # No periodic evaluation, only evaluate after training (speed up training)
    
    print("Training completed!")
    
    # Load best model
    checkpoint = torch.load('best_lightgcn_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("Final model evaluation...")
    model.eval()
    
    print("  Evaluating on validation set (recommendations)...")
    val_metrics = evaluate_model(
        model, validation_recommendations, user_id_map, movie_id_map, 
        graph_data, k=10, train_ratings=train_ratings,
        validation_type='recommendations'
    )
    
    print("  Evaluating on test set (test_ratings)...")
    test_metrics = evaluate_model(
        model, test_ratings, user_id_map, movie_id_map, 
        graph_data, k=10, train_ratings=train_ratings
    )
    
    recall = test_metrics.get('recall', 0)
    ndcg = test_metrics.get('ndcg', 0)
    precision = test_metrics.get('precision', 0)
    hr = test_metrics.get('hr', 0)
    
    print(f"\n" + "="*60)
    print(f"Training completed! Final performance metrics")
    print(f"="*60)
    print(f"\nTest set (test_ratings):")
    print(f"  • Recall@10: {recall:.4f}")
    print(f"  • Precision@10: {precision:.4f}")
    print(f"  • HR@10: {hr:.4f}")
    print(f"  • NDCG@10: {ndcg:.4f}")
    print(f"\nValidation set (recommendations):")
    print(f"  • Recall@10: {val_metrics.get('recall', 0):.4f}")
    print(f"  • Precision@10: {val_metrics.get('precision', 0):.4f}")
    print(f"  • HR@10: {val_metrics.get('hr', 0):.4f}")
    print(f"  • NDCG@10: {val_metrics.get('ndcg', 0):.4f}")
    
    # Save final embeddings
    print("Saving model embeddings...")
    with torch.no_grad():
        # Get all node embeddings
        all_embeddings = model.get_embeddings(graph_data.edge_index, graph_data.edge_weight)
        # Separate user and movie embeddings
        user_embeddings = all_embeddings[:len(user_id_map)]
        movie_embeddings = all_embeddings[len(user_id_map):]
    
    torch.save({
        'user_embeddings': user_embeddings.cpu(),
        'movie_embeddings': movie_embeddings.cpu(),
        'user_id_map': user_id_map,
        'movie_id_map': movie_id_map,
        'id_to_user': loader.id_to_user,
        'id_to_movie': loader.id_to_movie,
        'movies_df': loader.movies_df
    }, 'lightgcn_embeddings.pt')
    
    print("Training completed! Model and embeddings saved.")
    
    return {
        'model': model,
        'user_embeddings': user_embeddings,
        'movie_embeddings': movie_embeddings,
        'loader': loader,
        'recall': recall,
        'precision': precision,
        'hr': hr,
        'ndcg': ndcg,
        'val_recall': val_metrics.get('recall', 0),
        'val_precision': val_metrics.get('precision', 0),
        'val_hr': val_metrics.get('hr', 0),
        'val_ndcg': val_metrics.get('ndcg', 0)
    }

def main():
    """Main function"""
    import sys
    import io
    # Set UTF-8 encoding to avoid Windows console encoding issues
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Training configuration
    config = {
        'embedding_dim': 64,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 1024,
        'num_epochs': 100,
        'num_negatives': 1,
        'test_size': 0.2,
        'random_seed': 42,
        'patience': 10,
        'eval_interval': 999999  # Set to a large value, no evaluation during training (only at the end)
    }
    
    print("LightGCN + LLM Hybrid Recommendation System")
    print("=" * 60)
    print("Phase 1: Train LightGCN model")
    print("=" * 60)
    
    # Train model
    result = train_lightgcn(config)
    
    if result:
        print(f"\n" + "="*60)
        print(f"Final training results summary")
        print(f"="*60)
        print(f"\nTest set performance (ratings):")
        print(f"  • Recall@10: {result['recall']:.4f}")
        print(f"  • Precision@10: {result.get('precision', 0):.4f}")
        print(f"  • HR@10: {result.get('hr', 0):.4f}")
        print(f"  • NDCG@10: {result['ndcg']:.4f}")
        if 'val_recall' in result:
            print(f"\nValidation set performance (recommendations):")
            print(f"  • Recall@10: {result.get('val_recall', 0):.4f}")
            print(f"  • Precision@10: {result.get('val_precision', 0):.4f}")
            print(f"  • HR@10: {result.get('val_hr', 0):.4f}")
            print(f"  • NDCG@10: {result.get('val_ndcg', 0):.4f}")
        
        # Test recommendation function
        print(f"\nTesting recommendation function...")
        test_user_id = list(result['loader'].user_id_map.keys())[0]
        test_user_idx = result['loader'].user_id_map[test_user_id]
        
        # Get candidate movies
        candidates = result['loader'].get_candidate_movies(
            test_user_idx, 
            result['user_embeddings'], 
            result['movie_embeddings'], 
            top_k=10
        )
        
        print(f"Top-10 recommendations for user {test_user_id}:")
        for i, movie in enumerate(candidates[:10]):
            print(f"  {i+1}. {movie['title']} ({movie['genres']}) - Score: {movie['score']:.4f}")

if __name__ == '__main__':
    main()