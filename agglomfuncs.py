#!/usr/bin/env python3

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import entropy
import networkx as nx

def create_quality_scores(X: np.ndarray, y: np.ndarray, model_predictions: np.ndarray,
                         model_uncertainties: Optional[np.ndarray] = None,
                         minimize: bool = True,
                         uncertainty_weight: float = 0.2) -> np.ndarray:
    """Create quality scores for points based on model predictions and uncertainties

    Args:
        model_predictions: Raw GPR predictions or acquisition function scores.
        minimize: If True, lower predicted values → higher quality (for raw predictions
                  in a minimization task). If False, higher values → higher quality
                  (use this when passing acquisition scores like EI where higher = better).
        uncertainty_weight: Weight for uncertainty in the combined score.
    """
    # Normalize predictions to [0,1] using self-range
    pred_range = np.max(model_predictions) - np.min(model_predictions)
    if pred_range > 0:
        if minimize:
            # Lower prediction → higher quality score
            normalized_predictions = (np.max(model_predictions) - model_predictions) / pred_range
        else:
            # Higher prediction → higher quality score (e.g. for acquisition scores)
            normalized_predictions = (model_predictions - np.min(model_predictions)) / pred_range
    else:
        normalized_predictions = np.zeros_like(model_predictions)

    # Calculate diversity scores using minimum spanning tree
    dist_matrix = squareform(pdist(X))
    mst = minimum_spanning_tree(dist_matrix).toarray()
    diversity_scores = np.sum(mst > 0, axis=1)
    div_range = np.max(diversity_scores) - np.min(diversity_scores)
    if div_range > 0:
        diversity_scores = (diversity_scores - np.min(diversity_scores)) / div_range
    else:
        diversity_scores = np.zeros_like(diversity_scores, dtype=float)

    # Combine predictions and diversity
    if model_uncertainties is not None:
        # Normalize uncertainties
        unc_range = np.max(model_uncertainties) - np.min(model_uncertainties)
        if unc_range > 0:
            uncertainties = (model_uncertainties - np.min(model_uncertainties)) / unc_range
        else:
            uncertainties = np.zeros_like(model_uncertainties)
        # Weighted combination using configurable uncertainty_weight
        pred_div_weight = (1.0 - uncertainty_weight) / 2.0
        quality_scores = pred_div_weight * normalized_predictions + pred_div_weight * diversity_scores + uncertainty_weight * uncertainties
    else:
        quality_scores = 0.5 * normalized_predictions + 0.5 * diversity_scores

    return quality_scores

def kmeans_select(X: np.ndarray, quality_scores: np.ndarray, n_points: int) -> np.ndarray:
    """Select points using k-means clustering"""
    kmeans = KMeans(n_clusters=n_points, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    selected_indices = []
    for i in range(n_points):
        cluster_points = np.where(cluster_labels == i)[0]
        if len(cluster_points) > 0:
            # Select point with highest quality score in cluster
            cluster_qualities = quality_scores[cluster_points]
            best_in_cluster = cluster_points[np.argmax(cluster_qualities)]
            selected_indices.append(best_in_cluster)
    
    # If we have fewer points than requested, add highest quality unselected points
    if len(selected_indices) < n_points:
        unselected = list(set(range(len(X))) - set(selected_indices))
        unselected_qualities = quality_scores[unselected]
        n_remaining = n_points - len(selected_indices)
        additional = np.array(unselected)[np.argsort(unselected_qualities)[-n_remaining:]]
        selected_indices.extend(additional)
    
    return np.array(selected_indices)

def hybrid_select(X: np.ndarray, quality_scores: np.ndarray, distance_matrix: np.ndarray, 
                 n_points: int, quality_weight: float = 0.7) -> np.ndarray:
    """Hybrid selection combining quality and diversity"""
    selected_indices = []
    available_indices = list(range(len(X)))
    
    # Select first point with highest quality
    first_idx = np.argmax(quality_scores)
    selected_indices.append(first_idx)
    available_indices.remove(first_idx)
    
    # Iteratively select remaining points
    while len(selected_indices) < n_points and available_indices:
        # Calculate diversity score as minimum distance to selected points
        diversity_scores = np.min(distance_matrix[available_indices][:, selected_indices], axis=1)
        diversity_scores = (diversity_scores - np.min(diversity_scores)) / (np.max(diversity_scores) - np.min(diversity_scores))
        
        # Calculate combined score
        available_qualities = quality_scores[available_indices]
        combined_scores = quality_weight * available_qualities + (1 - quality_weight) * diversity_scores
        
        # Select point with highest combined score
        best_idx = available_indices[np.argmax(combined_scores)]
        selected_indices.append(best_idx)
        available_indices.remove(best_idx)
    
    return np.array(selected_indices)

def entropy_select(X: np.ndarray, quality_scores: np.ndarray, n_points: int) -> np.ndarray:
    """Select points using entropy-based diversity"""
    selected_indices = []
    available_indices = list(range(len(X)))
    
    # Select first point with highest quality
    first_idx = np.argmax(quality_scores)
    selected_indices.append(first_idx)
    available_indices.remove(first_idx)
    
    while len(selected_indices) < n_points and available_indices:
        # Calculate pairwise distances
        distances = cdist(X[available_indices], X[selected_indices])
        
        # Calculate entropy for each available point
        entropies = np.zeros(len(available_indices))
        for i in range(len(available_indices)):
            # Create probability distribution from distances
            probs = 1 / (distances[i] + 1e-10)
            probs = probs / np.sum(probs)
            entropies[i] = entropy(probs)
        
        # Normalize entropies
        entropies = (entropies - np.min(entropies)) / (np.max(entropies) - np.min(entropies))
        
        # Combine with quality scores
        available_qualities = quality_scores[available_indices]
        combined_scores = 0.7 * available_qualities + 0.3 * entropies
        
        # Select point with highest combined score
        best_idx = available_indices[np.argmax(combined_scores)]
        selected_indices.append(best_idx)
        available_indices.remove(best_idx)
    
    return np.array(selected_indices)

def graph_select(X: np.ndarray, quality_scores: np.ndarray, n_points: int) -> np.ndarray:
    """Select points using graph-based diversity"""
    # Create graph from distance matrix
    dist_matrix = squareform(pdist(X))
    threshold = np.mean(dist_matrix) + np.std(dist_matrix)
    adjacency = dist_matrix < threshold
    
    # Create networkx graph
    G = nx.from_numpy_array(adjacency)
    
    # Calculate centrality measures
    degree_centrality = np.array(list(nx.degree_centrality(G).values()))
    betweenness_centrality = np.array(list(nx.betweenness_centrality(G).values()))
    
    # Normalize centrality measures
    degree_centrality = (degree_centrality - np.min(degree_centrality)) / (np.max(degree_centrality) - np.min(degree_centrality))
    betweenness_centrality = (betweenness_centrality - np.min(betweenness_centrality)) / (np.max(betweenness_centrality) - np.min(betweenness_centrality))
    
    # Combine scores
    combined_scores = 0.4 * quality_scores + 0.3 * degree_centrality + 0.3 * betweenness_centrality
    
    # Select top points
    return np.argsort(combined_scores)[-n_points:]

def select_points(X: np.ndarray, 
                 quality_scores: np.ndarray, 
                 method: str = "hybrid", 
                 n_points: int = 10,
                 kernel_matrix: Optional[np.ndarray] = None,
                 config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Select points using specified method with configurable parameters"""
    if config is None:
        config = {
            "quality_weight": 0.7,
            "uncertainty_bonus": 0.2
        }
        
    methods = {
        "kmeans": lambda: kmeans_select(X, quality_scores, n_points),
        "hybrid": lambda: hybrid_select(X, quality_scores, cdist(X, X), n_points, config["quality_weight"]),
        "entropy": lambda: entropy_select(X, quality_scores, n_points),
        "graph": lambda: graph_select(X, quality_scores, n_points)
    }
    
    if method not in methods:
        raise ValueError(f"Unknown selection method: {method}. Available methods: {list(methods.keys())}")
    
    return methods[method]() 
