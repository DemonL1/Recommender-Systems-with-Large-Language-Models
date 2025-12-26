import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np

class LightGCNConv(MessagePassing):
    """LightGCN Convolution Layer"""
    
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr='add')
    
    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))
        
        # Calculate inverse square root of degree matrix
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize edge weights
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # Message passing
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class LightGCN(nn.Module):
    """LightGCN Model (supports BERT semantic features for cold start)"""
    
    def __init__(self, num_nodes, embedding_dim=64, num_layers=3, dropout=0.1, 
                 use_bert=False, movie_semantic_features=None, cold_start_movie_indices=None, 
                 num_users=0, fusion_weight=0.5, final_bert_weight=None):
        super(LightGCN, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_users = num_users
        
        # BERT semantic feature support (for cold start)
        self.use_bert = use_bert and movie_semantic_features is not None
        self.movie_semantic_features = movie_semantic_features  # Store BERT semantic vectors
        self.cold_start_movie_indices = cold_start_movie_indices or set()
        self.fusion_weight = fusion_weight  # Fusion weight between BERT and graph embeddings (initialization phase, 0-1)
        
        # Final fusion phase BERT weight (if None, automatically calculated based on cold start ratio)
        if final_bert_weight is None:
            # Automatically reduce final weight if cold start is rare
            # Assume cold start ratio affects final weight: lower ratio = lower weight
            cold_start_ratio = len(self.cold_start_movie_indices) / max(num_nodes - num_users, 1)
            # 0.15% cold start ratio should result in very low final weight (0.1-0.2)
            self.final_bert_weight = min(fusion_weight * 0.3, 0.2)  # Max 0.2, typically 0.1-0.15
        else:
            self.final_bert_weight = final_bert_weight
        
        # Initialize embeddings
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.reset_parameters()
        
        # Convolution layers
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
        
        # If using BERT, create projection layer (project BERT features to match graph embedding dimension)
        if self.use_bert and movie_semantic_features is not None:
            # Ensure BERT feature dimension matches embedding_dim
            bert_dim = movie_semantic_features.shape[1] if len(movie_semantic_features.shape) > 1 else embedding_dim
            if bert_dim != embedding_dim:
                print(f"  • Creating BERT projection layer: {bert_dim} -> {embedding_dim}")
                self.bert_projection = nn.Linear(bert_dim, embedding_dim)
                nn.init.xavier_uniform_(self.bert_projection.weight)
                nn.init.zeros_(self.bert_projection.bias)
            else:
                self.bert_projection = None
    
    def reset_parameters(self):
        """Reset parameters"""
        nn.init.normal_(self.embedding.weight, std=0.1)
    
    def forward(self, edge_index, edge_weight=None):
        """Forward pass (supports BERT semantic feature fusion)"""
        x = self.embedding.weight
        
        # If using BERT, fuse BERT semantic features for cold start items
        if self.use_bert and self.movie_semantic_features is not None:
            x = self._fuse_bert_features(x, edge_index.device)
        
        # Store embeddings from each layer
        embeddings = [x]
        
        # Pass through convolution layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings.append(x)
        
        # Average embeddings from all layers
        out = torch.stack(embeddings, dim=0).mean(dim=0)
        
        # For cold start items, fuse BERT features again in final output (enhance semantic information)
        if self.use_bert and self.movie_semantic_features is not None:
            out = self._fuse_bert_features_final(out, edge_index.device)
        
        return out
    
    def _fuse_bert_features(self, graph_embeddings, device):
        """Fuse BERT semantic features (for cold start item initialization, conservative fusion)"""
        if not self.use_bert or self.movie_semantic_features is None:
            return graph_embeddings
        
        # Skip fusion if fusion weight is very low or zero
        if self.fusion_weight <= 0.01:
            return graph_embeddings
        
        # Move BERT features to correct device
        bert_features = self.movie_semantic_features.to(device)
        
        # Project if dimensions don't match
        if self.bert_projection is not None:
            bert_features = self.bert_projection(bert_features)
        
        # Normalize BERT features to match graph embedding distribution
        bert_features = F.normalize(bert_features, p=2, dim=-1)
        
        # Create fused embeddings
        fused_embeddings = graph_embeddings.clone()
        
        # For cold start items, use BERT features (or fusion of BERT and graph embeddings)
        num_users = self.num_users if self.num_users > 0 else (len(graph_embeddings) - len(bert_features))
        
        for movie_idx in self.cold_start_movie_indices:
            if movie_idx >= num_users and movie_idx < num_users + len(bert_features):
                movie_idx_in_map = movie_idx - num_users
                if movie_idx_in_map < len(bert_features):
                    # Fusion strategy: determine fusion ratio based on fusion_weight
                    # fusion_weight=0: use only graph embeddings
                    # fusion_weight=1: use only BERT features
                    # For rare cold start cases, use low weight (0.1-0.2)
                    bert_feat = bert_features[movie_idx_in_map]
                    graph_feat = graph_embeddings[movie_idx]
                    
                    # Normalize graph embedding (match BERT features)
                    graph_feat_norm = F.normalize(graph_feat.unsqueeze(0), p=2, dim=-1).squeeze(0)
                    
                    # Weighted fusion (using normalized features)
                    fused_embeddings[movie_idx] = (
                        (1 - self.fusion_weight) * graph_feat_norm + 
                        self.fusion_weight * bert_feat
                    )
        
        return fused_embeddings
    
    def _fuse_bert_features_final(self, graph_embeddings, device):
        """Fuse BERT features in final output (conservative fusion to avoid interfering with normal movies)"""
        if not self.use_bert or self.movie_semantic_features is None:
            return graph_embeddings
        
        # Skip final fusion if final weight is very low or zero (let graph convolution learn on its own)
        if self.final_bert_weight <= 0.01:
            return graph_embeddings
        
        # For cold start items, conservatively fuse BERT features in final output
        bert_features = self.movie_semantic_features.to(device)
        
        # Project if dimensions don't match
        if self.bert_projection is not None:
            bert_features = self.bert_projection(bert_features)
        
        # Normalize BERT features and graph embeddings to match distributions
        bert_features = F.normalize(bert_features, p=2, dim=-1)
        
        fused_embeddings = graph_embeddings.clone()
        num_users = self.num_users if self.num_users > 0 else (len(graph_embeddings) - len(bert_features))
        
        # For cold start items, use lower BERT weight (avoid interference)
        for movie_idx in self.cold_start_movie_indices:
            if movie_idx >= num_users and movie_idx < num_users + len(bert_features):
                movie_idx_in_map = movie_idx - num_users
                if movie_idx_in_map < len(bert_features):
                    bert_feat = bert_features[movie_idx_in_map]
                    graph_feat = graph_embeddings[movie_idx]
                    
                    # Normalize graph embedding
                    graph_feat_norm = F.normalize(graph_feat.unsqueeze(0), p=2, dim=-1).squeeze(0)
                    
                    # Fuse with configurable low weight (should be very low for rare cold starts)
                    fused_embeddings[movie_idx] = (
                        (1 - self.final_bert_weight) * graph_feat_norm + 
                        self.final_bert_weight * bert_feat
                    )
        
        return fused_embeddings
    
    def encode_minibatch(self, users, items, edge_index, edge_weight):
        """Encode user and item embeddings (for mini-batch training)"""
        all_embeddings = self.forward(edge_index, edge_weight)
        user_emb = all_embeddings[users]
        item_emb = all_embeddings[items]
        return user_emb, item_emb
    
    def get_embeddings(self, edge_index, edge_weight=None):
        """Get embeddings for all nodes (supports BERT semantic features)"""
        return self.forward(edge_index, edge_weight)
    
    def get_item_embeddings(self, item_indices, edge_index, edge_weight=None, use_bert=True):
        """Get embeddings for specified items (supports using BERT for cold start items)"""
        all_embeddings = self.get_embeddings(edge_index, edge_weight)
        item_embeddings = all_embeddings[item_indices]
        
        # For cold start items, can use pure BERT features if needed
        if use_bert and self.use_bert and self.movie_semantic_features is not None:
            num_users = self.num_users if self.num_users > 0 else (len(all_embeddings) - len(self.movie_semantic_features))
            bert_features = self.movie_semantic_features.to(item_embeddings.device)
            
            if self.bert_projection is not None:
                bert_features = self.bert_projection(bert_features)
            
            for i, item_idx in enumerate(item_indices):
                if item_idx in self.cold_start_movie_indices:
                    if item_idx >= num_users and item_idx < num_users + len(bert_features):
                        movie_idx_in_map = item_idx - num_users
                        if movie_idx_in_map < len(bert_features):
                            # For cold start items, use pure BERT features (more accurate)
                            item_embeddings[i] = bert_features[movie_idx_in_map]
        
        return item_embeddings

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """BPR Loss Function"""
    pos_score = (user_emb * pos_item_emb).sum(dim=-1)
    neg_score = (user_emb * neg_item_emb).sum(dim=-1)
    loss = -F.logsigmoid(pos_score - neg_score).mean()
    return loss

def create_negative_samples(ratings_df, user_id_map, movie_id_map, num_negatives=1):
    """Create Negative Samples (Efficient Version)"""
    from tqdm import tqdm
    print("Creating negative samples...")
    
    # Use pandas groupby to build user-item interaction dictionary more efficiently
    print("  • Building user-item interaction dictionary...")
    user_pos_items = ratings_df.groupby('userId')['movieId'].apply(set).to_dict()
    print(f"  • Number of users: {len(user_pos_items):,}")
    
    # Create triplets (using vectorized operations)
    print("  • Generating training triplets...")
    triplets = []
    all_movie_ids = np.array(list(movie_id_map.keys()))
    num_users = len(user_id_map)
    
    # Show progress with tqdm
    for user_id, pos_items in tqdm(user_pos_items.items(), desc="  Generating negative samples", total=len(user_pos_items)):
        user_idx = user_id_map[user_id]
        
        # Fast calculation of candidate negative items
        pos_items_array = np.array(list(pos_items))
        # Use set difference operation (faster)
        candidate_neg_items = all_movie_ids[~np.isin(all_movie_ids, pos_items_array)]
        
        if len(candidate_neg_items) == 0:
            continue
        
        # Create negative samples for each positive sample
        for pos_movie_id in pos_items:
            pos_idx = movie_id_map[pos_movie_id] + num_users
            
            # Create negative samples for each positive sample
            for _ in range(num_negatives):
                # Randomly sample negative samples
                neg_movie_id = np.random.choice(candidate_neg_items)
                neg_idx = movie_id_map[neg_movie_id] + num_users
                
                triplets.append((user_idx, pos_idx, neg_idx))
    
    print(f"  • Created {len(triplets):,} training triplets")
    return triplets

class LightGCNDataset:
    """LightGCN Dataset Class"""
    
    def __init__(self, triplets, batch_size=1024):
        self.triplets = triplets
        self.batch_size = batch_size
        self.num_batches = len(triplets) // batch_size
    
    def __iter__(self):
        """Iterator"""
        # Randomly shuffle data
        indices = torch.randperm(len(self.triplets))
        
        for i in range(0, len(self.triplets), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_triplets = [self.triplets[idx] for idx in batch_indices]
            
            users = torch.LongTensor([t[0] for t in batch_triplets])
            pos_items = torch.LongTensor([t[1] for t in batch_triplets])
            neg_items = torch.LongTensor([t[2] for t in batch_triplets])
            
            yield users, pos_items, neg_items
    
    def __len__(self):
        return self.num_batches

def evaluate_model(model, test_ratings, user_id_map, movie_id_map, graph_data, id_to_movie=None, k=10, max_users=None, train_ratings=None, validation_type='ratings'):
    """Evaluate Model Performance (Optimized Version, Excludes Movies in Training Set)
    
    Args:
        model: Model instance
        test_ratings: Test/validation data (can be ratings or recommendations)
        user_id_map: User ID mapping dictionary
        movie_id_map: Movie ID mapping dictionary
        graph_data: Graph data object
        id_to_movie: Mapping from index to movie ID
        k: Top-K value for evaluation
        max_users: Maximum number of users to evaluate
        train_ratings: Training set ratings (used to exclude seen movies)
        validation_type: Validation type, 'ratings' or 'recommendations'
    """
    from tqdm import tqdm
    
    # Show information based on validation type
    if validation_type == 'recommendations':
        print("Evaluating model performance (validation set: recommendations)...")
    else:
        print("Evaluating model performance (test set: ratings)...")
    
    # Create reverse mapping from movie_id_map if id_to_movie not provided
    if id_to_movie is None:
        id_to_movie = {idx: movie_id for movie_id, idx in movie_id_map.items()}
    
    # Build user training set movie sets (for exclusion)
    user_train_items = {}
    if train_ratings is not None:
        for user_id in user_id_map.keys():
            user_train_items[user_id] = set(train_ratings[train_ratings['userId'] == user_id]['movieId'])
        print(f"  • Loaded training set information (for excluding seen movies)")
    
    model.eval()
    with torch.no_grad():
        # Get all embeddings (pass graph_data parameters)
        print("  • Generating embeddings...")
        all_embeddings = model.get_embeddings(graph_data.edge_index, graph_data.edge_weight)
        user_embeddings = all_embeddings[:len(user_id_map)]
        movie_embeddings = all_embeddings[len(user_id_map):]
        
        # Get test users (limit number to speed up evaluation)
        test_users = test_ratings['userId'].unique()
        if max_users and len(test_users) > max_users:
            print(f"  • Limiting evaluation to {max_users} users (total users: {len(test_users)})")
            test_users = test_users[:max_users]
        
        # Calculate multiple metrics
        recalls = []
        precisions = []
        hit_rates = []  # HR@K
        ndcgs = []
        
        print(f"  • Evaluating {len(test_users)} users...")
        for user_id in tqdm(test_users, desc="  Evaluation progress"):
            # Get positive samples for user in test set
            user_test_items = set(test_ratings[test_ratings['userId'] == user_id]['movieId'])
            
            if len(user_test_items) == 0:
                continue
            
            if user_id not in user_id_map:
                continue
                
            user_idx = user_id_map[user_id]
            user_emb = user_embeddings[user_idx]
            
            # Calculate scores for all movies
            scores = torch.matmul(user_emb, movie_embeddings.T)
            
            # Exclude movies seen in training set
            seen_items = user_train_items.get(user_id, set())
            
            # Get scores and indices for all movies, exclude seen ones
            valid_indices = []
            valid_scores = []
            for idx in range(len(movie_embeddings)):
                movie_id = id_to_movie.get(idx)
                if movie_id is not None and movie_id not in seen_items:
                    valid_indices.append(idx)
                    valid_scores.append(scores[idx].item())
            
            if len(valid_indices) == 0:
                continue
            
            # Get top-k recommendations (from unseen movies)
            valid_scores = torch.tensor(valid_scores)
            top_k_indices_in_valid = valid_scores.argsort(descending=True)[:k]
            top_k_indices = [valid_indices[idx.item()] for idx in top_k_indices_in_valid]
            top_k_movies = [id_to_movie.get(idx, None) for idx in top_k_indices]
            top_k_movies = [m for m in top_k_movies if m is not None]
            
            # Calculate number of hits
            hits = len(set(top_k_movies) & user_test_items)
            
            # 1. Recall@K: number of hits / total positive samples in test set
            recall = hits / len(user_test_items) if len(user_test_items) > 0 else 0
            recalls.append(recall)
            
            # 2. Precision@K: number of hits / K (ratio of hits in Top-K)
            precision = hits / k if k > 0 else 0
            precisions.append(precision)
            
            # 3. Hit Rate@K (HR@K): whether at least one hit (0 or 1)
            hit_rate = 1.0 if hits > 0 else 0.0
            hit_rates.append(hit_rate)
            
            # 4. NDCG@K
            dcg = 0
            idcg = 0
            for i, movie_id in enumerate(top_k_movies):
                if movie_id in user_test_items:
                    dcg += 1 / np.log2(i + 2)
            
            for i in range(min(len(user_test_items), k)):
                idcg += 1 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
        
        # Calculate average values
        avg_recall = np.mean(recalls) if recalls else 0
        avg_precision = np.mean(precisions) if precisions else 0
        avg_hr = np.mean(hit_rates) if hit_rates else 0
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0
        
        # Print all metrics
        print(f"  • Recall@{k}: {avg_recall:.4f} (hits/test set size)")
        print(f"  • Precision@{k}: {avg_precision:.4f} (hits/{k})")
        print(f"  • HR@{k} (Hit Rate): {avg_hr:.4f} (ratio of users with hits)")
        print(f"  • NDCG@{k}: {avg_ndcg:.4f} (ranking quality)")
        
        return {
            'recall': avg_recall,
            'precision': avg_precision,
            'hr': avg_hr,
            'ndcg': avg_ndcg
        }

if __name__ == '__main__':
    # Test model
    print("Testing LightGCN model...")
    
    # Create test data
    num_nodes = 1000
    embedding_dim = 64
    num_layers = 3
    
    # Create model
    model = LightGCN(num_nodes, embedding_dim, num_layers)
    
    # Create test graph
    edge_index = torch.randint(0, num_nodes, (2, 1000))
    edge_weight = torch.rand(1000)
    
    # Test forward pass
    embeddings = model(edge_index, edge_weight)
    print(f"Output embedding shape: {embeddings.shape}")
    
    # Test BPR loss
    user_emb = torch.randn(10, embedding_dim)
    pos_item_emb = torch.randn(10, embedding_dim)
    neg_item_emb = torch.randn(10, embedding_dim)
    
    loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
    print(f"BPR loss: {loss.item():.4f}")