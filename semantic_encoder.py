#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Encoder Module
Semantic feature extraction based on Hugging Face Transformers
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class SemanticEncoder(nn.Module):
    """Transformer-based Semantic Encoder"""
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 embedding_dim: int = 64, device: str = "auto"):
        super(SemanticEncoder, self).__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Auto device selection
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing semantic encoder: {model_name}")
        print(f"Using device: {self.device}")
        
        try:
            # Initialize BERT model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.to(self.device)
            
            # Get BERT output dimension
            bert_dim = self.bert_model.config.hidden_size
            
            # Projection layer to map BERT output to target dimension
            self.projection = nn.Linear(bert_dim, embedding_dim)
            self.projection.to(self.device)
            
            # Initialize projection layer weights
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            
            print(f"Semantic encoder initialized successfully")
            print(f"  • BERT dimension: {bert_dim}")
            print(f"  • Target dimension: {embedding_dim}")
            
        except Exception as e:
            print(f"Semantic encoder initialization failed: {e}")
            # Use fallback model
            self._init_fallback_model()
    
    def _init_fallback_model(self):
        """Initialize fallback model"""
        print("Using fallback SentenceTransformer model...")
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_model.to(self.device)
            self.use_sentence_model = True
            print("Fallback model initialized successfully")
        except Exception as e:
            print(f"Fallback model initialization failed: {e}")
            self.use_sentence_model = False
    
    def encode_movie_features(self, movie_titles: List[str], 
                            genres: List[str] = None) -> torch.Tensor:
        """Encode movie semantic features"""
        if not movie_titles:
            return torch.zeros((0, self.embedding_dim), device=self.device)
        
        try:
            if hasattr(self, 'use_sentence_model') and self.use_sentence_model:
                return self._encode_with_sentence_transformer(movie_titles, genres)
            else:
                return self._encode_with_bert(movie_titles, genres)
        except Exception as e:
            print(f"Semantic encoding failed: {e}")
            # Return zero vectors as fallback
            return torch.zeros((len(movie_titles), self.embedding_dim), device=self.device)
    
    def _encode_with_bert(self, movie_titles: List[str], genres: List[str] = None) -> torch.Tensor:
        """Encode with BERT"""
        # Build movie description text
        descriptions = []
        for i, title in enumerate(movie_titles):
            if genres and i < len(genres):
                desc = f"Movie: {title}. Genres: {genres[i]}"
            else:
                desc = f"Movie: {title}"
            descriptions.append(desc)
        
        # Batch encoding
        inputs = self.tokenizer(
            descriptions, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token representation
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            
        # Project to target dimension
        projected = self.projection(embeddings)
        return projected
    
    def _encode_with_sentence_transformer(self, movie_titles: List[str], 
                                        genres: List[str] = None) -> torch.Tensor:
        """Encode with SentenceTransformer"""
        # Build description text
        descriptions = []
        for i, title in enumerate(movie_titles):
            if genres and i < len(genres):
                desc = f"Movie: {title}. Genres: {genres[i]}"
            else:
                desc = f"Movie: {title}"
            descriptions.append(desc)
        
        # Encode
        embeddings = self.sentence_model.encode(descriptions, convert_to_tensor=True)
        embeddings = embeddings.to(self.device)
        
        # Project to target dimension
        projected = self.projection(embeddings)
        return projected
    
    def encode_user_preferences(self, user_history: List[str], 
                              max_length: int = 256) -> torch.Tensor:
        """Encode user preference semantic features"""
        if not user_history:
            return torch.zeros(self.embedding_dim, device=self.device)
        
        try:
            if hasattr(self, 'use_sentence_model') and self.use_sentence_model:
                return self._encode_user_with_sentence_transformer(user_history)
            else:
                return self._encode_user_with_bert(user_history, max_length)
        except Exception as e:
            print(f"User preference encoding failed: {e}")
            return torch.zeros(self.embedding_dim, device=self.device)
    
    def _encode_user_with_bert(self, user_history: List[str], max_length: int) -> torch.Tensor:
        """Encode user preferences with BERT"""
        # Build user preference description
        preference_text = f"User preferences: {', '.join(user_history[:10])}"  # Limit length
        
        inputs = self.tokenizer(
            preference_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        projected = self.projection(embeddings)
        return projected.squeeze(0)
    
    def _encode_user_with_sentence_transformer(self, user_history: List[str]) -> torch.Tensor:
        """Encode user preferences with SentenceTransformer"""
        # Build preference text
        preference_text = f"User preferences: {', '.join(user_history[:10])}"
        
        embeddings = self.sentence_model.encode([preference_text], convert_to_tensor=True)
        embeddings = embeddings.to(self.device)
        
        projected = self.projection(embeddings)
        return projected.squeeze(0)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Batch encode texts"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if hasattr(self, 'use_sentence_model') and self.use_sentence_model:
                batch_embeddings = self.sentence_model.encode(
                    batch_texts, convert_to_tensor=True
                ).to(self.device)
                batch_embeddings = self.projection(batch_embeddings)
            else:
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                    batch_embeddings = self.projection(batch_embeddings)
            
            all_embeddings.append(batch_embeddings)
        
        return torch.cat(all_embeddings, dim=0)

class MovieSemanticProcessor:
    """Movie Semantic Feature Processor"""
    
    def __init__(self, semantic_encoder: SemanticEncoder):
        self.encoder = semantic_encoder
        self.movie_cache = {}  # Cache encoding results
    
    def process_movie_catalog(self, movies_df) -> torch.Tensor:
        """Process movie catalog to generate semantic features"""
        print("Processing movie catalog semantic features...")
        
        movie_titles = movies_df['title'].tolist()
        movie_genres = movies_df['genres'].tolist()
        
        # Check cache
        cache_key = f"movies_{len(movie_titles)}"
        if cache_key in self.movie_cache:
            print("Using cached movie semantic features")
            return self.movie_cache[cache_key]
        
        # Encode movie features
        movie_embeddings = self.encoder.encode_movie_features(movie_titles, movie_genres)
        
        # Cache results
        self.movie_cache[cache_key] = movie_embeddings
        
        print(f"Movie semantic feature processing completed: {movie_embeddings.shape}")
        return movie_embeddings
    
    def process_user_history(self, user_ratings_df, movies_df) -> Dict[int, torch.Tensor]:
        """Process user history to generate user preference semantic features"""
        print("Processing user preference semantic features...")
        
        user_embeddings = {}
        unique_users = user_ratings_df['userId'].unique()
        
        for user_id in unique_users:
            # Get user's historical movies
            user_movies = user_ratings_df[user_ratings_df['userId'] == user_id]
            user_movie_titles = []
            
            for _, row in user_movies.iterrows():
                movie_id = row['movieId']
                movie_info = movies_df[movies_df['movieId'] == movie_id]
                if not movie_info.empty:
                    title = movie_info.iloc[0]['title']
                    user_movie_titles.append(title)
            
            # Encode user preferences
            if user_movie_titles:
                user_emb = self.encoder.encode_user_preferences(user_movie_titles)
                user_embeddings[user_id] = user_emb
        
        print(f"User preference semantic feature processing completed: {len(user_embeddings)} users")
        return user_embeddings

def main():
    """Test semantic encoder"""
    print("Testing semantic encoder...")
    
    # Create encoder
    encoder = SemanticEncoder(embedding_dim=64)
    
    # Test movie encoding
    test_titles = ["The Matrix", "Inception", "Interstellar"]
    test_genres = ["Sci-Fi|Action", "Sci-Fi|Thriller", "Sci-Fi|Drama"]
    
    movie_embeddings = encoder.encode_movie_features(test_titles, test_genres)
    print(f"Movie embedding shape: {movie_embeddings.shape}")
    
    # Test user preference encoding
    test_history = ["The Matrix", "Inception", "Blade Runner"]
    user_embedding = encoder.encode_user_preferences(test_history)
    print(f"User embedding shape: {user_embedding.shape}")
    
    print("Semantic encoder test completed successfully")

if __name__ == '__main__':
    main()




