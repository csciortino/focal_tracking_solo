import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import hdbscan
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


class IndividualIdentifier:
    def __init__(self, method='hdbscan', **kwargs):
        self.method = method
        self.scaler = StandardScaler()
        self.reducer = None
        self.clusterer = None
        self.labels_ = None
        self.features_ = None
        self.track_ids_ = None
        
        # Default parameters for different methods
        if method == 'hdbscan':
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=kwargs.get('min_cluster_size', 3),
                min_samples=kwargs.get('min_samples', 2),
                metric='euclidean'
            )
        elif method == 'dbscan':
            self.clusterer = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 3)
            )
    
    def fit_predict(self, tracklet_features: Dict[int, np.ndarray]) -> Dict[int, int]:
        # Filter out empty features
        valid_tracklets = {tid: feat for tid, feat in tracklet_features.items() 
                          if len(feat) > 0}
        
        if len(valid_tracklets) == 0:
            return {}
            
        # Prepare data
        track_ids = list(valid_tracklets.keys())
        features = np.array(list(valid_tracklets.values()))
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Dimensionality reduction if needed
        if features_scaled.shape[1] > 50:
            self.reducer = UMAP(n_components=min(20, len(track_ids)-1), 
                               random_state=42)
            features_reduced = self.reducer.fit_transform(features_scaled)
        else:
            features_reduced = features_scaled
            
        # Clustering
        cluster_labels = self.clusterer.fit_predict(features_reduced)
        
        # Store results
        self.features_ = features_reduced
        self.track_ids_ = track_ids
        self.labels_ = cluster_labels
        
        # Create track_id to individual_id mapping
        track_to_individual = {}
        for track_id, label in zip(track_ids, cluster_labels):
            if label >= 0:  # -1 indicates noise/outlier
                track_to_individual[track_id] = label
            else:
                # Assign unique ID to noise points
                track_to_individual[track_id] = -1
                
        return track_to_individual
    
    def visualize_clusters(self, save_path: str = None) -> plt.Figure:
        if self.features_ is None or self.labels_ is None:
            raise ValueError("Must call fit_predict first")
            
        # Use t-SNE for visualization if features are high-dimensional
        if self.features_.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.features_)-1))
            features_2d = tsne.fit_transform(self.features_)
        else:
            features_2d = self.features_
            
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot colored by cluster
        unique_labels = set(self.labels_)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'black'
                marker = 'x'
                alpha = 0.5
            else:
                marker = 'o'
                alpha = 0.8
                
            mask = self.labels_ == label
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[color], marker=marker, alpha=alpha, s=50,
                       label=f'Individual {label}' if label >= 0 else 'Noise')
        
        ax1.set_title('Proposed Individual Clusters')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cluster statistics
        cluster_counts = pd.Series(self.labels_).value_counts().sort_index()
        cluster_counts.plot(kind='bar', ax=ax2)
        ax2.set_title('Tracklets per Proposed Individual')
        ax2.set_xlabel('Individual ID')
        ax2.set_ylabel('Number of Tracklets')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def get_cluster_statistics(self) -> pd.DataFrame:
        if self.labels_ is None or self.track_ids_ is None:
            raise ValueError("Must call fit_predict first")
            
        stats = []
        for track_id, label in zip(self.track_ids_, self.labels_):
            stats.append({
                'track_id': track_id,
                'individual_id': label,
                'is_noise': label == -1
            })
            
        df = pd.DataFrame(stats)
        
        summary_stats = df.groupby('individual_id').agg({
            'track_id': 'count',
            'is_noise': 'first'
        }).rename(columns={'track_id': 'num_tracklets'})
        
        return summary_stats
    
    def merge_similar_clusters(self, similarity_threshold: float = 0.8) -> Dict[int, int]:
        if self.features_ is None or self.labels_ is None:
            return {}
            
        # Compute cluster centroids
        unique_labels = [l for l in set(self.labels_) if l >= 0]
        centroids = {}
        
        for label in unique_labels:
            mask = self.labels_ == label
            centroids[label] = np.mean(self.features_[mask], axis=0)
            
        # Compute pairwise similarities between centroids
        merge_map = {}
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                similarity = np.corrcoef(centroids[label1], centroids[label2])[0, 1]
                if similarity > similarity_threshold:
                    # Merge smaller cluster into larger one
                    count1 = np.sum(self.labels_ == label1)
                    count2 = np.sum(self.labels_ == label2)
                    if count1 >= count2:
                        merge_map[label2] = label1
                    else:
                        merge_map[label1] = label2
                        
        return merge_map