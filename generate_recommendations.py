#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommendation Experience Script
Generate recommendations for users using a trained LightGCN model
"""

import torch
import pandas as pd
from data_loader import MovieLensDataLoader
from model import LightGCN

def load_model_and_data():
    """Load model and data"""
    print("Loading model and data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    loader = MovieLensDataLoader()
    if not loader.load_preprocessed_data():
        print("Data loading failed")
        return None, None, None, None
    
    # Load model
    checkpoint = torch.load('best_lightgcn_model.pth', map_location=device)
    config = checkpoint['config']
    
    model = LightGCN(
        num_nodes=loader.graph_data.num_nodes,
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model and data loaded successfully!")
    
    return model, loader, device, config

def generate_recommendations(model, loader, device, user_id, top_k=10):
    """Generate recommendations for a user"""
    print(f"\nGenerating recommendations for user {user_id}...")
    
    # Check if user exists
    if user_id not in loader.user_id_map:
        print(f"User {user_id} does not exist")
        return None
    
    # Get movies rated by the user
    user_ratings = loader.ratings_df[loader.ratings_df['userId'] == user_id]
    rated_movies = set(user_ratings['movieId'].tolist())
    
    print(f"  • Number of movies rated by user: {len(rated_movies)}")
    if len(user_ratings) > 0:
        avg_rating = user_ratings['rating'].mean()
        print(f"  • Average rating: {avg_rating:.2f}")
    
    # Generate embeddings
    with torch.no_grad():
        graph_data = loader.graph_data.to(device)
        all_embeddings = model.get_embeddings(graph_data.edge_index, graph_data.edge_weight)
        user_embeddings = all_embeddings[:len(loader.user_id_map)]
        movie_embeddings = all_embeddings[len(loader.user_id_map):]
    
    # Get user embedding
    user_idx = loader.user_id_map[user_id]
    user_emb = user_embeddings[user_idx]
    
    # Calculate scores for all movies
    scores = torch.matmul(user_emb, movie_embeddings.T)
    
    # Get top-k recommendations (exclude rated movies)
    all_scores = scores.cpu().numpy()
    movie_indices = list(range(len(loader.movie_id_map)))
    
    # Create list of (movie index, score)
    movie_scores = []
    for idx, score in enumerate(all_scores):
        movie_id = loader.id_to_movie[idx]
        if movie_id not in rated_movies:  # Exclude rated movies
            movie_scores.append((movie_id, score))
    
    # Sort by score
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_movies = movie_scores[:top_k]
    
    # Get movie details
    recommendations = []
    for movie_id, score in top_k_movies:
        movie_info = loader.movies_df[loader.movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            recommendations.append({
                'movieId': movie_id,
                'title': movie_info.iloc[0]['title'],
                'genres': movie_info.iloc[0]['genres'],
                'score': score
            })
    
    return recommendations

def display_recommendations(recommendations, user_id):
    """Display recommendation results"""
    print(f"\nTop-{len(recommendations)} Recommendations for User {user_id}:")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Genre: {rec['genres']}")
        print(f"   Recommendation Score: {rec['score']:.4f}")
        print(f"   Movie ID: {rec['movieId']}")
    
    print("\n" + "=" * 80)

def show_user_history(loader, user_id, top_n=5):
    """Show user's rating history"""
    user_ratings = loader.ratings_df[loader.ratings_df['userId'] == user_id]
    
    if len(user_ratings) == 0:
        print(f"User {user_id} has no rating history")
        return
    
    # Merge movie information
    merged = user_ratings.merge(loader.movies_df, on='movieId')
    top_ratings = merged.nlargest(top_n, 'rating')
    
    print(f"\nUser {user_id}'s Highly Rated Movies:")
    print("-" * 80)
    for _, movie in top_ratings.iterrows():
        print(f"  • {movie['title']} ({movie['genres']}) - Rating: {movie['rating']:.1f}")

def main():
    """Main function"""
    print("LightGCN Recommendation System Experience")
    print("=" * 80)
    
    # Load model and data
    model, loader, device, config = load_model_and_data()
    if model is None:
        return
    
    # Get some active users as examples
    user_rating_counts = loader.ratings_df['userId'].value_counts()
    active_users = user_rating_counts.head(5).index.tolist()
    
    print(f"\nSelecting active users for recommendation demonstration...")
    print(f"Active user list: {active_users[:5]}")
    
    # Generate recommendations for each user
    for user_id in active_users[:3]:  # Demonstrate for first 3 users only
        print("\n" + "=" * 80)
        
        # Show user history
        show_user_history(loader, user_id, top_n=3)
        
        # Generate recommendations
        recommendations = generate_recommendations(model, loader, device, user_id, top_k=10)
        
        if recommendations:
            display_recommendations(recommendations, user_id)
    
    print("\n" + "=" * 80)
    print("Recommendation generation completed!")
    print("\nTips:")
    print("  • Modify the user_id in the code to generate recommendations for specific users")
    print("  • Higher recommendation scores indicate the model thinks the user is more likely to like the movie")
    print("  • Rated movies are automatically excluded from recommendations")

if __name__ == '__main__':
    main()




