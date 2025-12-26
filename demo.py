#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MovieLens Belief Dataset Demonstration
Showcase data analysis and basic recommendation functionality
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and analyze data"""
    print("MovieLens Belief Dataset Demonstration")
    print("=" * 60)
    
    try:
        # Load data
        print("Loading data...")
        movies_df = pd.read_csv('data/movies.csv')
        ratings_df = pd.read_csv('data/user_rating_history.csv')
        beliefs_df = pd.read_csv('data/belief_data.csv')
        
        print("Data loaded successfully!")
        print(f"  • Movies: {len(movies_df):,}")
        print(f"  • User ratings: {len(ratings_df):,}")
        print(f"  • User beliefs: {len(beliefs_df):,}")
        
        # Data overview
        print("\nData Overview:")
        print(f"  • Number of users: {ratings_df['userId'].nunique():,}")
        print(f"  • Number of movies: {ratings_df['movieId'].nunique():,}")
        print(f"  • Average rating: {ratings_df['rating'].mean():.2f}")
        print(f"  • Rating standard deviation: {ratings_df['rating'].std():.2f}")
        
        # Movie genre analysis
        print("\nMovie Genre Analysis:")
        all_genres = []
        for genres in movies_df['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        print("Popular movie genres:")
        for i, (genre, count) in enumerate(genre_counts.head(10).items(), 1):
            print(f"  {i:2d}. {genre}: {count:,} movies")
        
        # User activity analysis
        print("\nUser Activity Analysis:")
        user_rating_counts = ratings_df['userId'].value_counts()
        print(f"  • Maximum ratings by a single user: {user_rating_counts.max():,}")
        print(f"  • Average ratings per user: {user_rating_counts.mean():.1f}")
        print(f"  • Median ratings per user: {user_rating_counts.median():.1f}")
        
        # Belief data analysis
        print("\nBelief Data Analysis:")
        beliefs_clean = beliefs_df.dropna(subset=['userPredictRating'])
        print(f"  • Valid belief records: {len(beliefs_clean):,}")
        print(f"  • Average predicted rating: {beliefs_clean['userPredictRating'].mean():.2f}")
        print(f"  • Predicted rating standard deviation: {beliefs_clean['userPredictRating'].std():.2f}")
        
        return movies_df, ratings_df, beliefs_df
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None, None, None

def simple_recommendation_demo(movies_df, ratings_df):
    """Simple recommendation demonstration"""
    print("\nSimple Recommendation Demonstration:")
    
    # Select most active users
    user_rating_counts = ratings_df['userId'].value_counts()
    top_users = user_rating_counts.head(5).index
    
    for i, user_id in enumerate(top_users, 1):
        print(f"\nUser {user_id} (Top {i} Active User):")
        
        # Get user ratings
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        user_ratings = user_ratings.merge(movies_df, on='movieId')
        
        # Show user preferences
        high_rated = user_ratings[user_ratings['rating'] >= 4.0]
        if len(high_rated) > 0:
            print(f"  • Highly Rated Movies ({len(high_rated)} movies):")
            for _, movie in high_rated.head(3).iterrows():
                print(f"    - {movie['title']} ({movie['genres']}) - Rating: {movie['rating']}")
        
        # Analyze user preferred genres
        all_genres = []
        for genres in user_ratings['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        if len(genre_counts) > 0:
            print(f"  • Preferred Genres: {', '.join(genre_counts.head(3).index)}")
        
        # Simple recommendation: movies based on user's preferred genre
        if len(genre_counts) > 0:
            top_genre = genre_counts.index[0]
            genre_movies = movies_df[movies_df['genres'].str.contains(top_genre, na=False)]
            
            # Exclude movies already rated by user
            rated_movies = set(user_ratings['movieId'])
            unrated_movies = genre_movies[~genre_movies['movieId'].isin(rated_movies)]
            
            if len(unrated_movies) > 0:
                print(f"  • Recommendations (Based on Preferred Genre '{top_genre}'):")
                for _, movie in unrated_movies.head(3).iterrows():
                    print(f"    - {movie['title']} ({movie['genres']})")

def belief_analysis_demo(beliefs_df, ratings_df):
    """Belief data analysis demonstration"""
    print("\nBelief Data Analysis Demonstration:")
    
    # Find users with both beliefs and actual ratings
    belief_users = beliefs_df['userId'].unique()
    rating_users = ratings_df['userId'].unique()
    common_users = set(belief_users) & set(rating_users)
    
    print(f"  • Users with both beliefs and ratings: {len(common_users):,}")
    
    if len(common_users) > 0:
        # Analyze a sample user
        sample_user = list(common_users)[0]
        print(f"\nSample User {sample_user} - Belief vs Actual Ratings:")
        
        user_beliefs = beliefs_df[beliefs_df['userId'] == sample_user]
        user_ratings = ratings_df[ratings_df['userId'] == sample_user]
        
        # Find common movies
        belief_movies = set(user_beliefs['movieId'])
        rating_movies = set(user_ratings['movieId'])
        common_movies = belief_movies & rating_movies
        
        if len(common_movies) > 0:
            print(f"  • Number of common movies: {len(common_movies)}")
            
            # Compare first few movies
            for movie_id in list(common_movies)[:3]:
                belief_rating = user_beliefs[user_beliefs['movieId'] == movie_id]['userPredictRating'].iloc[0]
                actual_rating = user_ratings[user_ratings['movieId'] == movie_id]['rating'].iloc[0]
                error = abs(belief_rating - actual_rating)
                
                print(f"    - Movie {movie_id}: Predicted {belief_rating:.1f} vs Actual {actual_rating:.1f} (Error: {error:.1f})")

def main():
    """Main function"""
    # Load and analyze data
    movies_df, ratings_df, beliefs_df = load_and_analyze_data()
    
    if movies_df is None:
        return
    
    # Simple recommendation demo
    simple_recommendation_demo(movies_df, ratings_df)
    
    # Belief data analysis demo
    belief_analysis_demo(beliefs_df, ratings_df)
    
    print("\nDemonstration completed!")
    print("\nNext Steps:")
    print("  1. Install additional dependencies to run the full system")
    print("  2. Use a machine with more memory to process the complete dataset")
    print("  3. Consider using cloud services for large-scale computations")
    print("  4. Explore applications of belief data in recommendation systems")

if __name__ == '__main__':
    main()