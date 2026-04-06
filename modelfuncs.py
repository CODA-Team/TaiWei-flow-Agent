#!/usr/bin/env python3

import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.stats import norm, qmc
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def latin_hypercube(n_points, n_dims):
    """Generate Latin Hypercube samples"""
    sampler = qmc.LatinHypercube(d=n_dims)
    return sampler.random(n=n_points)

def parse_llm_response(llm_response: str) -> Dict[str, Any]:
    """Parse the LLM's response into structured recommendations"""
    # This would need to be adapted based on your actual LLM's output format
    # For now, assuming a structured response that can be parsed
    try:
        # Example parsing - adapt based on actual LLM response format
        recommendations = {
            "kernel": llm_response.split("Kernel choice:")[1].split("\n")[0].strip(),
            "preprocessing": llm_response.split("Preprocessing:")[1].split("\n")[0].strip(),
            "acquisition": llm_response.split("Acquisition:")[1].split("\n")[0].strip(),
            "missing_values": llm_response.split("Missing values:")[1].split("\n")[0].strip(),
            "surrogate": llm_response.split("Surrogate:")[1].split("\n")[0].strip()
        }
        return recommendations
    except:
        # Fallback to default recommendations
        return {
            "kernel": "matern",
            "preprocessing": "standard",
            "acquisition": "ei",
            "missing_values": "surrogate_fill",
            "surrogate": "weighted_combination"
        }

def create_kernel(kernel_spec: str, input_dim: int) -> Any:
    """Create a kernel based on recommendations"""
    # Define length scale bounds
    ls_bounds = (1e-2, 1e2)  # More conservative bounds
    
    base_kernels = {
        "rbf": RBF(length_scale=[1.0] * input_dim, length_scale_bounds=[ls_bounds] * input_dim),
        "matern": Matern(length_scale=[1.0] * input_dim, length_scale_bounds=[ls_bounds] * input_dim, nu=2.5),
        "rational": RationalQuadratic(length_scale=[1.0] * input_dim, length_scale_bounds=[ls_bounds] * input_dim, alpha=1.0),
        "composite": Matern(length_scale=[1.0] * input_dim, length_scale_bounds=[ls_bounds] * input_dim, nu=2.5)
    }
    
    # Default to Matern kernel if specification not recognized
    base_kernel = base_kernels.get(kernel_spec.lower(), base_kernels["matern"])
    
    # Add noise term with adaptive scaling
    noise_kernel = WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * base_kernel + noise_kernel
    
    return kernel

def create_preprocessor(preprocess_spec: str) -> Any:
    """Create a preprocessor based on recommendations"""
    preprocessors = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "none": None
    }
    return preprocessors.get(preprocess_spec.lower(), preprocessors["standard"])

def create_acquisition_function(acq_spec: str) -> Callable:
    """Create an acquisition function based on recommendations"""
    def expected_improvement(mean, std, best_f):
        with np.errstate(divide='warn'):
            # Add small epsilon to std to prevent division by zero
            std = np.maximum(std, 1e-9)
            z = (best_f - mean) / std
            return std * (z * norm.cdf(z) + norm.pdf(z))
    
    def upper_confidence_bound(mean, std, best_f, beta=2.0):
        # Lower confidence bound for minimization, negated so higher = better
        return -(mean - beta * std)
    
    def probability_improvement(mean, std, best_f, xi=0.01):
        # Add exploration bonus xi
        std = np.maximum(std, 1e-9)
        z = (best_f - mean - xi) / std
        return norm.cdf(z)
    
    def augmented_ei(mean, std, best_f, xi=0.01, alpha=0.5):
        # Combine EI with PI for better exploration
        ei = expected_improvement(mean, std, best_f)
        pi = probability_improvement(mean, std, best_f, xi)
        return alpha * ei + (1 - alpha) * pi
    
    acquisitions = {
        "ei": expected_improvement,
        "ucb": upper_confidence_bound,
        "pi": probability_improvement,
        "augmented_ei": augmented_ei
    }
    return acquisitions.get(acq_spec.lower(), acquisitions["augmented_ei"])

def create_model(X, y, noise_level=1e-6, kernel_type="matern", preprocessing="standard", feature_weights=None):
    """Create and configure a Gaussian Process model with improved features"""
    n_features = X.shape[1]

    # Handle case with no or insufficient data points
    if len(X) < 2:
        print("Warning: Insufficient data points for modeling. Creating dummy model.")
        # Return a simple model that will encourage exploration
        kernel = 1.0 * RBF(length_scale=[1.0] * n_features)
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=42
        )
        # Fit with dummy data to enable predictions
        X_dummy = np.vstack([X, X + 0.1])  # Add slightly offset point
        y_dummy = np.array([y[0], y[0]])   # Duplicate the single y value
        model.fit(X_dummy, y_dummy)
        model.scaler_ = create_preprocessor(preprocessing) or StandardScaler()
        model.scaler_.fit(X_dummy)
        return model

    # Scale the input features using the configured preprocessor
    scaler = create_preprocessor(preprocessing)
    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
    else:
        # preprocessing="none": no scaling, but store an identity-like scaler for predict_with_model
        scaler = StandardScaler(with_mean=False, with_std=False)
        X_scaled = scaler.fit_transform(X)
    
    # Use provided feature weights if available, otherwise compute from data
    if feature_weights is not None and len(feature_weights) == n_features:
        correlations = np.array(feature_weights, dtype=float)
    else:
        correlations = np.zeros(n_features)
        if len(y) > 1:  # Only compute correlations if we have enough points
            correlations = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(n_features)])
    important_features = correlations > np.mean(correlations)

    # Create adaptive length scales based on feature importance
    length_scales = np.ones(n_features)
    length_scales[important_features] = 0.5  # Shorter length scales for important features
    
    # Create adaptive bounds based on data scale
    ls_bounds = (1e-2, 1e2)  # More conservative bounds
    
    # Create base kernel
    if kernel_type == "matern":
        kernel = Matern(length_scale=length_scales, length_scale_bounds=[ls_bounds] * n_features, nu=2.5)
    elif kernel_type == "rbf":
        kernel = RBF(length_scale=length_scales, length_scale_bounds=[ls_bounds] * n_features)
    elif kernel_type == "rational":
        kernel = RationalQuadratic(length_scale=length_scales, length_scale_bounds=[ls_bounds] * n_features, alpha=1.0)
    else:
        # For composite, use Matern with optimized parameters
        kernel = Matern(length_scale=length_scales, length_scale_bounds=[ls_bounds] * n_features, nu=2.5)
    
    # Add noise term with adaptive scaling
    y_std = np.std(y) if len(y) > 1 else 1.0
    noise_level = max(noise_level, 1e-6 * y_std)
    kernel = kernel + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10 * y_std, 1e-2 * y_std))
    
    # Create GP model with increased optimization
    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,  # Increased from 5
        normalize_y=True,
        random_state=42
    )
    
    # Fit the model
    try:
        model.fit(X_scaled, y)
    except Exception as e:
        print(f"Warning: Error fitting model: {str(e)}. Creating simpler model.")
        # Fall back to simpler kernel if fitting fails
        simple_kernel = 1.0 * RBF(length_scale=[1.0] * n_features)
        model = GaussianProcessRegressor(
            kernel=simple_kernel,
            normalize_y=True,
            random_state=42
        )
        model.fit(X_scaled, y)
    
    # Store scaler for future predictions
    model.scaler_ = scaler
    
    return model

def handle_surrogate_data(X, y, surrogate_values):
    """Process and combine true and surrogate data with improved uncertainty handling"""
    if surrogate_values is None:
        return X, y
    
    # Identify valid and missing data points
    valid_mask = ~np.isnan(y)
    missing_mask = ~valid_mask
    
    # Calculate uncertainty in surrogate values based on data range
    if np.any(valid_mask):
        y_range = np.max(y[valid_mask]) - np.min(y[valid_mask])
        surrogate_uncertainty = 0.1 * y_range  # 10% of range
    else:
        surrogate_uncertainty = 1.0
    
    # Create weighted combination
    y_combined = y.copy()
    y_combined[missing_mask] = surrogate_values[missing_mask]
    
    # Add uncertainty information
    uncertainty = np.zeros_like(y)
    uncertainty[missing_mask] = surrogate_uncertainty
    
    return X, y_combined, uncertainty

def expected_improvement(mu, std, y_best, xi=0.01):
    """Calculate expected improvement acquisition function with improved numerical stability"""
    mu = mu.reshape(-1, 1)
    sigma = std.reshape(-1, 1)
    
    # Add small epsilon to prevent division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate improvement
    imp = y_best - mu - xi
    
    # Calculate Z-score
    Z = imp / sigma
    
    # Calculate expected improvement
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0
    
    return ei.ravel()

def evaluate_timing_model(context: dict) -> dict:
    """Evaluate timing model performance with improved analysis"""
    model_results = {
        'performance': {},
        'suggestions': []
    }
    
    # Extract data
    log_data = context.get('log_data', {})
    metrics = context.get('metrics', {})
    model_recommendations = context.get('model_recommendations', {})
    
    # Check data availability
    if not metrics.get('objectives'):
        model_results['suggestions'].append(
            "No timing data available yet. Using default model configuration.")
        return model_results
        
    # Analyze timing correlations with improved metrics
    correlations = metrics.get('correlations', {})
    if correlations.get('real_vs_surrogate'):
        real = np.asarray([x['real'] for x in correlations['real_vs_surrogate']], dtype=float)
        surrogate = np.asarray([x['surrogate'] for x in correlations['real_vs_surrogate']], dtype=float)
        if len(real) > 1 and np.std(real) > 0 and np.std(surrogate) > 0:
            surrogate_correlation = float(np.corrcoef(real, surrogate)[0, 1])
            model_results['performance']['surrogate_correlation'] = surrogate_correlation

            abs_gap = float(np.mean(np.abs(real - surrogate)))
            model_results['performance']['surrogate_abs_gap'] = abs_gap
            
            if surrogate_correlation > 0.8:
                model_results['suggestions'].append(
                    "Strong surrogate correlation - increase surrogate weight")
            elif surrogate_correlation < 0.5:
                model_results['suggestions'].append(
                    "Weak surrogate correlation - reduce surrogate weight")

            if abs_gap > max(0.05, 0.5 * np.std(real)):
                model_results['suggestions'].append(
                    "Timing surrogate gap is large; validate more runs with final timing metrics.")
    
    # Add model recommendations from structure analysis
    if model_recommendations:
        model_results['recommendations'] = model_recommendations
        rec_kernel = model_recommendations.get('kernel_type', '')
        if rec_kernel and rec_kernel != 'matern':
            model_results['suggestions'].append(
                f"Data analysis suggests '{rec_kernel}' kernel may fit better than default Matern.")
        if model_recommendations.get('needs_local_models', False):
            model_results['suggestions'].append(
                "Cluster differences detected; consider local surrogate models.")
        if model_recommendations.get('use_feature_weights', False):
            model_results['suggestions'].append(
                "Feature importance varies significantly; weighted length scales recommended.")

    return model_results

def evaluate_wirelength_model(context: dict) -> dict:
    """Evaluate wirelength model performance with improved analysis"""
    model_results = {
        'performance': {},
        'suggestions': []
    }
    
    # Extract data
    log_data = context.get('log_data', {})
    metrics = context.get('metrics', {})
    model_recommendations = context.get('model_recommendations', {})
    
    # Check data availability
    if not metrics.get('objectives'):
        model_results['suggestions'].append(
            "No wirelength data available yet. Using default model configuration.")
        return model_results
        
    # Analyze wirelength progression with improved metrics
    wl_progression = metrics.get('wirelength_progression', [])
    if wl_progression:
        # Calculate average ratio between final and CTS wirelength
        ratios = [p['final'] / p['cts'] for p in wl_progression if p['cts'] > 0]
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            model_results['performance']['cts_to_final_ratio'] = avg_ratio
            
            if avg_ratio > 1.5:
                model_results['suggestions'].append(
                    "Large increase in wirelength after CTS - reduce surrogate weight")
            elif avg_ratio < 1.2:
                model_results['suggestions'].append(
                    "CTS wirelength is good predictor - increase surrogate weight")
    
    # Add model recommendations from structure analysis
    if model_recommendations:
        model_results['recommendations'] = model_recommendations
        rec_kernel = model_recommendations.get('kernel_type', '')
        if rec_kernel and rec_kernel != 'matern':
            model_results['suggestions'].append(
                f"Data analysis suggests '{rec_kernel}' kernel may fit better than default Matern.")
        if model_recommendations.get('needs_local_models', False):
            model_results['suggestions'].append(
                "Cluster differences detected; consider local surrogate models.")
        if model_recommendations.get('use_feature_weights', False):
            model_results['suggestions'].append(
                "Feature importance varies significantly; weighted length scales recommended.")

    return model_results

def evaluate_combo_model(context: dict) -> dict:
    """Evaluate combined ECP/WL objective behavior using normalized combo labels."""
    model_results = {
        'performance': {},
        'suggestions': []
    }

    metrics = context.get('metrics', {})
    model_recommendations = context.get('model_recommendations', {})

    objectives = metrics.get('objectives', [])
    surrogates = metrics.get('surrogates', [])
    if not objectives:
        model_results['suggestions'].append(
            "No COMBO objective data available yet. Using default model configuration.")
        return model_results

    valid_objectives = np.asarray(objectives, dtype=float)
    model_results['performance']['combo_mean'] = float(np.mean(valid_objectives))
    model_results['performance']['combo_std'] = float(np.std(valid_objectives))
    model_results['performance']['combo_best'] = float(np.min(valid_objectives))
    model_results['performance']['n_objective_samples'] = int(len(valid_objectives))

    # Since the COMBO label is a baseline-normalized minimization objective,
    # negative values indicate improvement relative to baseline.
    improved_ratio = float(np.mean(valid_objectives < 0))
    model_results['performance']['improved_ratio'] = improved_ratio
    if improved_ratio > 0.5:
        model_results['suggestions'].append(
            "More than half of COMBO samples beat baseline; tighten local search around strong regions.")
    elif improved_ratio < 0.1:
        model_results['suggestions'].append(
            "Few COMBO samples beat baseline; increase exploration and revisit parameter ranges.")

    if surrogates:
        valid_surrogates = np.asarray(surrogates, dtype=float)
        model_results['performance']['combo_surrogate_mean'] = float(np.mean(valid_surrogates))
        model_results['performance']['combo_surrogate_std'] = float(np.std(valid_surrogates))

    correlations = metrics.get('correlations', {})
    if correlations.get('real_vs_surrogate'):
        real = np.asarray([x['real'] for x in correlations['real_vs_surrogate']], dtype=float)
        surrogate = np.asarray([x['surrogate'] for x in correlations['real_vs_surrogate']], dtype=float)
        if len(real) > 1 and np.std(real) > 0 and np.std(surrogate) > 0:
            surrogate_corr = float(np.corrcoef(real, surrogate)[0, 1])
            model_results['performance']['surrogate_correlation'] = surrogate_corr

            abs_gap = float(np.mean(np.abs(real - surrogate)))
            model_results['performance']['surrogate_abs_gap'] = abs_gap

            if surrogate_corr > 0.8:
                model_results['suggestions'].append(
                    "COMBO surrogate tracks final objective well; trust proxy-guided exploitation more.")
            elif surrogate_corr < 0.5:
                model_results['suggestions'].append(
                    "COMBO surrogate is weakly aligned with final objective; emphasize exploration and final-metric validation.")

            if abs_gap > max(0.05, 0.5 * np.std(real)):
                model_results['suggestions'].append(
                    "The COMBO surrogate gap is large; avoid over-committing to CTS-driven signals.")

    distribution = metrics.get('distribution_analysis', {})
    y_missing_ratio = distribution.get('y_missing_ratio')
    if y_missing_ratio is not None:
        model_results['performance']['y_missing_ratio'] = float(y_missing_ratio)
        if y_missing_ratio > 0.3:
            model_results['suggestions'].append(
                "Many final COMBO labels are missing; surrogate fallback is active often, so diversify candidates.")

    if model_recommendations:
        model_results['recommendations'] = model_recommendations
        rec_kernel = model_recommendations.get('kernel_type', '')
        if rec_kernel and rec_kernel != 'matern':
            model_results['suggestions'].append(
                f"Data analysis suggests '{rec_kernel}' kernel may fit better than default Matern.")
        if model_recommendations.get('needs_local_models', False):
            model_results['suggestions'].append(
                "Cluster differences detected; consider local surrogate models.")
        if model_recommendations.get('use_feature_weights', False):
            model_results['suggestions'].append(
                "Feature importance varies significantly; weighted length scales recommended.")

    return model_results

def predict_with_model(model, X_new):
    """Make predictions with proper scaling"""
    X_scaled = model.scaler_.transform(X_new)
    return model.predict(X_scaled, return_std=True) 
