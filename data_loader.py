import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optionally import BERT-related modules
try:
    from semantic_encoder import SemanticEncoder, MovieSemanticProcessor
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("BERT module not found, BERT-based semantic features will be disabled.")

class MovieLensDataLoader:
    """
    Data loader and graph constructor for the MovieLens belief dataset. Handles raw data reading, ID mapping, heterogeneous graph construction, and (optionally) feature encoding for cold-start items using BERT-based semantic features.
    """
    
    def __init__(self, data_dir='data', use_bert=True, bert_model='bert-base-uncased'):
        self.data_dir = data_dir
        self.movies_df = None
        self.ratings_df = None
        self.beliefs_df = None
        self.recommendations_df = None
        self.elicitation_df = None
        
        # Mapping dictionaries
        self.user_id_map = {}
        self.movie_id_map = {}
        self.id_to_user = {}
        self.id_to_movie = {}
        
        # Graph data
        self.graph_data = None
        
        # BERT semantic features (for cold-start)
        self.use_bert = use_bert and BERT_AVAILABLE
        self.bert_model_name = bert_model
        self.semantic_encoder = None
        self.movie_semantic_features = None  # Stores the BERT semantic vectors for all movies
        self.cold_start_movie_indices = set()  # Indices set for cold-start items
        
    def load_data(self):
        """
        Load all relevant CSV data files for the experiment.
        """
        print("Loading data files ...")
        
        try:
            self.movies_df = pd.read_csv(f'{self.data_dir}/movies.csv')
            self.ratings_df = pd.read_csv(f'{self.data_dir}/user_rating_history.csv')
            self.beliefs_df = pd.read_csv(f'{self.data_dir}/belief_data.csv')
            self.recommendations_df = pd.read_csv(f'{self.data_dir}/user_recommendation_history.csv')
            self.elicitation_df = pd.read_csv(f'{self.data_dir}/movie_elicitation_set.csv')
            
            print("All data files loaded successfully!")
            print(f"  • Movies: {len(self.movies_df):,}")
            print(f"  • Ratings: {len(self.ratings_df):,}")
            print(f"  • Beliefs: {len(self.beliefs_df):,}")
            print(f"  • Recommendations: {len(self.recommendations_df):,}")
            
            return True
            
        except Exception as e:
            print(f"Data loading failed: {e}")
            return False
    
    def create_mappings(self):
        """
        Create user and movie ID-to-index bidirectional mappings, required for graph node representation and model input.
        """
        print("Creating ID mappings ...")
        
        # Get all user IDs
        all_user_ids = set()
        all_user_ids.update(self.ratings_df['userId'].unique())
        all_user_ids.update(self.beliefs_df['userId'].unique())
        all_user_ids.update(self.recommendations_df['userId'].unique())
        
        # Get all movie IDs
        all_movie_ids = set()
        all_movie_ids.update(self.ratings_df['movieId'].unique())
        all_movie_ids.update(self.beliefs_df['movieId'].unique())
        all_movie_ids.update(self.recommendations_df['movieId'].unique())
        all_movie_ids.update(self.elicitation_df['movieId'].unique())
        
        # Create mappings
        sorted_user_ids = sorted(all_user_ids)
        sorted_movie_ids = sorted(all_movie_ids)
        
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(sorted_user_ids)}
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(sorted_movie_ids)}
        
        self.id_to_user = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.id_to_movie = {idx: movie_id for movie_id, idx in self.movie_id_map.items()}
        
        print(f"  • Number of users: {len(self.user_id_map):,}")
        print(f"  • Number of movies: {len(self.movie_id_map):,}")
        
        return self.user_id_map, self.movie_id_map
    
    def build_graph(self, use_ratings=True, use_recommendations=True, use_beliefs=False):
        """
        Build a torch_geometric Graph from ratings, recommendations, and optionally beliefs, with weighted heterogeneous edges.
        """
        print("Constructing interaction graph ...")
        
        edge_sources = []
        edge_targets = []
        edge_weights = []
        
        # Add rating interactions (vectorized)
        if use_ratings:
            print("  • Adding rating interactions...")
            # Filter out NaN, negative, and invalid values
            ratings_clean = self.ratings_df.dropna(subset=['rating']).copy()
            ratings_clean = ratings_clean[(ratings_clean['rating'] >= 0) & (ratings_clean['rating'] <= 5)]
            print(f"  • Valid rating records after filtering: {len(ratings_clean):,} (original: {len(self.ratings_df):,})")
            
            # Use vectorized operations to improve speed
            user_indices = ratings_clean['userId'].map(self.user_id_map).values
            movie_indices = ratings_clean['movieId'].map(self.movie_id_map).values + len(self.user_id_map)
            ratings = ratings_clean['rating'].values
            
            # Filter out mapping failures (NaN)
            valid_mask = ~(np.isnan(user_indices) | np.isnan(movie_indices) | np.isnan(ratings))
            
            edge_sources.extend(user_indices[valid_mask].astype(int).tolist())
            edge_targets.extend(movie_indices[valid_mask].astype(int).tolist())
            edge_weights.extend(ratings[valid_mask].tolist())
        
        # Add recommendation interactions (vectorized)
        if use_recommendations:
            print("  • Adding recommendation interactions...")
            # Use vectorized operations
            user_indices = self.recommendations_df['userId'].map(self.user_id_map).values
            movie_indices = self.recommendations_df['movieId'].map(self.movie_id_map).values + len(self.user_id_map)
            
            # Filter out mapping failures
            valid_mask = ~(np.isnan(user_indices) | np.isnan(movie_indices))
            
            edge_sources.extend(user_indices[valid_mask].astype(int).tolist())
            edge_targets.extend(movie_indices[valid_mask].astype(int).tolist())
            edge_weights.extend([1.0] * valid_mask.sum())  # Recommendation edge weight is set to 1
        
        # Add belief interactions (optional)
        if use_beliefs:
            print("  • Adding belief interactions...")
            beliefs_clean = self.beliefs_df.dropna(subset=['userPredictRating']).copy()
            print(f"  • Valid belief records: {len(beliefs_clean):,}")
            
            # Use vectorized operations to process belief data
            user_indices = beliefs_clean['userId'].map(self.user_id_map).values
            movie_indices = beliefs_clean['movieId'].map(self.movie_id_map).values + len(self.user_id_map)
            predict_ratings = beliefs_clean['userPredictRating'].values
            
            # Filter out mapping failures
            valid_mask = ~(np.isnan(user_indices) | np.isnan(movie_indices) | np.isnan(predict_ratings))
            
            user_indices_valid = user_indices[valid_mask].astype(int)
            movie_indices_valid = movie_indices[valid_mask].astype(int)
            predict_ratings_valid = predict_ratings[valid_mask]
            
            # If certainty field exists, use certainty as weight
            if 'userCertainty' in beliefs_clean.columns:
                certainties = beliefs_clean['userCertainty'].values
                certainties_valid = certainties[valid_mask]
                # Normalize certainty to 0-1 range (original is 1-5)
                certainties_normalized = certainties_valid / 5.0
                # Edge weight = predicted rating * certainty weight
                belief_weights = predict_ratings_valid * (0.5 + 0.5 * certainties_normalized)  # Higher certainty = higher weight
                print(f"  • Using certainty weighting (certainty range: {certainties_valid.min():.1f}-{certainties_valid.max():.1f})")
            else:
                # If no certainty, use predicted rating directly
                belief_weights = predict_ratings_valid
                print(f"  • Certainty field not found, using predicted rating as weight")
            
            edge_sources.extend(user_indices_valid.tolist())
            edge_targets.extend(movie_indices_valid.tolist())
            edge_weights.extend(belief_weights.tolist())
            
            print(f"  • Successfully added {len(user_indices_valid):,} belief interactions")
        
        # Create edge index and weights
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # Check and clean NaN values
        nan_mask = torch.isnan(edge_weight)
        if nan_mask.any():
            print(f"  Warning: Found {nan_mask.sum().item()} NaN weights, filtered out")
            valid_mask = ~nan_mask
            edge_index = edge_index[:, valid_mask]
            edge_weight = edge_weight[valid_mask]
        
        # Check Inf values
        inf_mask = torch.isinf(edge_weight)
        if inf_mask.any():
            print(f"  Warning: Found {inf_mask.sum().item()} Inf weights, filtered out")
            valid_mask = ~inf_mask
            edge_index = edge_index[:, valid_mask]
            edge_weight = edge_weight[valid_mask]
        
        # Create graph data
        num_users = len(self.user_id_map)
        num_movies = len(self.movie_id_map)
        num_nodes = num_users + num_movies
        
        self.graph_data = Data(
            edge_index=edge_index, 
            edge_weight=edge_weight, 
            num_nodes=num_nodes
        )
        self.graph_data.num_users = num_users
        self.graph_data.num_movies = num_movies
        
        print(f"  • Total nodes: {num_nodes:,}")
        print(f"  • Total edges: {len(edge_sources):,}")
        
        # Identify cold-start items (movies without interaction history)
        if self.use_bert:
            self._identify_cold_start_items()
            self._generate_bert_features()
        
        return self.graph_data
    
    def _identify_cold_start_items(self):
        """Identify cold-start items (movies without interaction history)"""
        print("Identifying cold-start items...")
        
        # Get set of movie IDs with interactions
        interacted_movie_ids = set()
        interacted_movie_ids.update(self.ratings_df['movieId'].unique())
        interacted_movie_ids.update(self.recommendations_df['movieId'].unique())
        if self.beliefs_df is not None:
            interacted_movie_ids.update(self.beliefs_df['movieId'].unique())
        
        # Find movies with no interactions among all movies
        all_movie_ids = set(self.movie_id_map.keys())
        cold_start_movie_ids = all_movie_ids - interacted_movie_ids
        
        # Convert to indices
        self.cold_start_movie_indices = {
            self.movie_id_map[movie_id] + len(self.user_id_map) 
            for movie_id in cold_start_movie_ids 
            if movie_id in self.movie_id_map
        }
        
        print(f"  • Total movies: {len(all_movie_ids):,}")
        print(f"  • Movies with interactions: {len(interacted_movie_ids):,}")
        print(f"  • Cold-start movies: {len(cold_start_movie_ids):,} ({len(cold_start_movie_ids)/len(all_movie_ids)*100:.1f}%)")
    
    def _generate_bert_features(self):
        """Generate BERT semantic features for all movies"""
        if not self.use_bert or self.movies_df is None:
            return
        
        print("Generating BERT semantic features (for cold start)...")
        
        try:
            # Initialize BERT encoder
            if self.semantic_encoder is None:
                self.semantic_encoder = SemanticEncoder(
                    model_name=self.bert_model_name,
                    embedding_dim=64,  # Match LightGCN's embedding_dim
                    device='auto'
                )
            
            # Get all movie titles and genres
            movie_titles = self.movies_df['title'].tolist()
            movie_genres = self.movies_df['genres'].fillna('').tolist()
            
            # Generate BERT semantic vectors in batches
            print(f"  • Generating BERT semantic vectors for {len(movie_titles):,} movies...")
            batch_size = 32
            all_embeddings = []
            
            progress_bar = tqdm(
                range(0, len(movie_titles), batch_size),
                desc="  • BERT encoding progress",
                total=(len(movie_titles) + batch_size - 1) // batch_size,
                ncols=100
            )
            
            for i in progress_bar:
                batch_titles = movie_titles[i:i+batch_size]
                batch_genres = movie_genres[i:i+batch_size]
                
                # Encode with BERT
                batch_embeddings = self.semantic_encoder.encode_movie_features(
                    batch_titles, batch_genres
                )
                all_embeddings.append(batch_embeddings.cpu())  # Move to CPU to save GPU memory
            
            # Combine all embeddings
            self.movie_semantic_features = torch.cat(all_embeddings, dim=0)
            
            print(f"BERT semantic feature generation completed: {self.movie_semantic_features.shape}")
            print(f"  • Feature dimension: {self.movie_semantic_features.shape[1]}")
            
        except Exception as e:
            print(f"BERT feature generation failed: {e}")
            print("  BERT semantic features will not be used")
            self.use_bert = False
            self.movie_semantic_features = None
    
    def get_movie_semantic_feature(self, movie_idx):
        """Get BERT semantic feature for specified movie (for cold start)"""
        if not self.use_bert or self.movie_semantic_features is None:
            return None
        
        # Convert graph index to movie index
        num_users = len(self.user_id_map)
        if movie_idx < num_users:
            return None  # Not a movie node
        
        movie_idx_in_map = movie_idx - num_users
        if movie_idx_in_map >= len(self.movie_semantic_features):
            return None
        
        return self.movie_semantic_features[movie_idx_in_map]
    
    def is_cold_start_item(self, movie_idx):
        """Check if movie is a cold-start item"""
        return movie_idx in self.cold_start_movie_indices
    
    def get_user_profile(self, user_id, top_k=10):
        """Generate text profile for specified user"""
        if user_id not in self.user_id_map:
            return f"User {user_id} not found in the dataset."
        
        # Get user's highly rated movies
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        if len(user_ratings) == 0:
            return f"User {user_id} has no rating history."
        
        # Merge movie information
        merged = user_ratings.merge(self.movies_df, on='movieId')
        
        # Sort by rating and take top_k
        top_movies = merged.nlargest(top_k, 'rating')
        
        # Generate user profile text
        movie_descriptions = []
        for _, row in top_movies.iterrows():
            title = row['title']
            genres = row['genres']
            rating = row['rating']
            movie_descriptions.append(f"'{title}' ({genres}) - rated {rating:.1f}")
        
        profile_text = f"User {user_id} has highly rated movies such as: {', '.join(movie_descriptions)}."
        
        # Add user preference analysis
        all_genres = []
        for genres in merged['genres']:
            all_genres.extend(genres.split('|'))
        genre_counts = pd.Series(all_genres).value_counts()
        top_genres = genre_counts.head(3).index.tolist()
        
        if top_genres:
            profile_text += f" The user shows preference for {', '.join(top_genres)} genres."
        
        return profile_text
    
    def get_candidate_movies(self, user_idx, user_embeddings, movie_embeddings, top_k=100):
        """Generate LightGCN Top-K candidate list for user"""
        if user_idx >= len(user_embeddings):
            return []
        
        # Ensure embeddings are on CPU (if not already)
        if isinstance(user_embeddings, torch.Tensor) and user_embeddings.is_cuda:
            user_embeddings = user_embeddings.cpu()
        if isinstance(movie_embeddings, torch.Tensor) and movie_embeddings.is_cuda:
            movie_embeddings = movie_embeddings.cpu()
        
        user_emb = user_embeddings[user_idx] if isinstance(user_embeddings, torch.Tensor) else torch.tensor(user_embeddings[user_idx])
        if not isinstance(movie_embeddings, torch.Tensor):
            movie_embeddings = torch.tensor(movie_embeddings)
        
        # Calculate scores (dot product)
        scores = torch.matmul(user_emb, movie_embeddings.T)
        
        # Get top-k indices and scores
        top_k_values, top_indices = torch.topk(scores, k=min(top_k, len(scores)), largest=True)
        
        candidate_movies_info = []
        for i, idx in enumerate(top_indices):
            # Convert tensor index to Python int
            idx_int = idx.item() if isinstance(idx, torch.Tensor) else int(idx)
            
            # Check if index is valid
            if idx_int not in self.id_to_movie:
                continue
                
            movie_id = self.id_to_movie[idx_int]
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            
            if len(movie_row) > 0:
                movie_info = movie_row.iloc[0]
                score_value = top_k_values[i].item() if isinstance(top_k_values[i], torch.Tensor) else top_k_values[i]
                candidate_movies_info.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'score': float(score_value)
                })
        
        return candidate_movies_info
    
    def save_preprocessed_data(self, filename='preprocessed_data.pt'):
        """Save preprocessed data"""
        print(f"Saving preprocessed data to {filename}...")
        
        data_dict = {
            'graph': self.graph_data,
            'user_map': self.user_id_map,
            'movie_map': self.movie_id_map,
            'id_to_user': self.id_to_user,
            'id_to_movie': self.id_to_movie,
            'movies_df': self.movies_df,
            'ratings_df': self.ratings_df,
            'beliefs_df': self.beliefs_df,
            'recommendations_df': self.recommendations_df,
            'use_bert': self.use_bert,
            'cold_start_movie_indices': self.cold_start_movie_indices,
            'movie_semantic_features': self.movie_semantic_features if self.use_bert else None
        }
        
        torch.save(data_dict, filename)
        print("Data saved successfully!")
    
    def load_preprocessed_data(self, filename='preprocessed_data.pt'):
        """Load preprocessed data"""
        print(f"Loading preprocessed data from {filename}...")
        
        try:
            data_dict = torch.load(filename)
            self.graph_data = data_dict['graph']
            self.user_id_map = data_dict['user_map']
            self.movie_id_map = data_dict['movie_map']
            self.id_to_user = data_dict['id_to_user']
            self.id_to_movie = data_dict['id_to_movie']
            self.movies_df = data_dict['movies_df']
            self.ratings_df = data_dict['ratings_df']
            self.beliefs_df = data_dict['beliefs_df']
            self.recommendations_df = data_dict['recommendations_df']
            
            # Load BERT-related data (if exists)
            if 'use_bert' in data_dict:
                self.use_bert = data_dict['use_bert']
                self.cold_start_movie_indices = data_dict.get('cold_start_movie_indices', set())
                self.movie_semantic_features = data_dict.get('movie_semantic_features', None)
                if self.use_bert and self.movie_semantic_features is not None:
                    print(f"  • Loaded BERT semantic features: {self.movie_semantic_features.shape}")
                    print(f"  • Number of cold-start movies: {len(self.cold_start_movie_indices):,}")
            
            print("Preprocessed data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Preprocessed data loading failed: {e}")
            return False

def main():
    """Main function - Data preprocessing pipeline"""
    print("MovieLens Data Preprocessing")
    print("=" * 50)
    
    # Create data loader
    loader = MovieLensDataLoader()
    
    # Load data
    if not loader.load_data():
        return
    
    # Create mappings
    loader.create_mappings()
    
    # Build graph
    loader.build_graph(use_ratings=True, use_recommendations=True, use_beliefs=True)
    
    # Save preprocessed data
    loader.save_preprocessed_data()
    
    # Test user profile generation
    test_user_id = list(loader.user_id_map.keys())[0]
    profile = loader.get_user_profile(test_user_id)
    print(f"\nTest User Profile (User {test_user_id}):")
    print(profile)

if __name__ == '__main__':
    main()