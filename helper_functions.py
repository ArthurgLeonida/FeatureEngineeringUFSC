import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

############################ DEPENDENCE MEASURES ############################

def mutual_information_from_scratch(x, y, bins=10):
    """
    Implement Mutual Information using histogram method
    MI(X,Y) = H(X) + H(Y) - H(X,Y)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    def entropy(data, bins):
        hist, _ = np.histogram(data, bins=bins, density=False)
        total = hist.sum()
        if total == 0:
            return 0.0
        probs = hist / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def joint_entropy(x, y, bins):
        hist, _, _ = np.histogram2d(x, y, bins=bins)
        total = hist.sum()
        if total == 0:
            return 0.0
        probs = hist / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    h_x = entropy(x, bins)
    h_y = entropy(y, bins)
    h_xy = joint_entropy(x, y, bins)
    return h_x + h_y - h_xy

def rbf_kernel(x, y=None, sigma=None):
    """
    Radial Basis Function (Gaussian) kernel
    k(x,y) = exp(-||x-y||²/(2σ²))
    Memory-efficient version for large datasets.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    if y is None:
        y = x
    else:
        y = np.asarray(y, dtype=float).reshape(-1, 1)

    if sigma is None or sigma <= 0:
        # Use subsample for sigma estimation to avoid memory issues
        max_samples = 5000
        if x.shape[0] > max_samples:
            indices = np.random.choice(x.shape[0], max_samples, replace=False)
            sample_data = x[indices]
        else:
            sample_data = x
            
        if sample_data.shape[0] > 1:
            distances = pdist(sample_data, metric='euclidean')
            distances = distances[distances > 0]
            if distances.size:
                sigma = np.median(distances)
        if sigma is None or sigma <= 0 or np.isnan(sigma):
            sigma = 1.0

    # Correct broadcasting: x[:, None, :] - y[None, :, :]
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.exp(-dist_sq / (2 * sigma ** 2))

def center_kernel_matrix(K):
    """
    Center the kernel matrix (remove mean similarities)
    More efficient implementation without explicit H matrix creation
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if n == 0:
        return K
    
    # More efficient centering: K - row_means - col_means + total_mean
    row_means = np.mean(K, axis=1, keepdims=True)
    col_means = np.mean(K, axis=0, keepdims=True)
    total_mean = np.mean(K)
    
    K_centered = K - row_means - col_means + total_mean
    return K_centered

def hsic_from_scratch(x, y, sigma_x=None, sigma_y=None, max_samples=5000):
    """
    Implement HSIC (Hilbert-Schmidt Independence Criterion)
    Measures dependence using kernel similarities.
    
    For large datasets (>max_samples), uses random subsampling to avoid memory issues.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = x.shape[0]

    if n < 2:
        return 0.0
    
    # Subsample for large datasets to avoid memory errors
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        x = x[indices]
        y = y[indices]
        n = max_samples
        
    # Create kernel matrices
    K_x = rbf_kernel(x, sigma_x)
    K_y = rbf_kernel(y, sigma_y)
    
    # Center the kernel matrices
    K_x_centered = center_kernel_matrix(K_x)
    K_y_centered = center_kernel_matrix(K_y)
    
    # HSIC = trace(K_x_centered @ K_y_centered) / (n-1)^2
    hsic_value = np.trace(K_x_centered @ K_y_centered) / ((n-1)**2)
    
    return hsic_value

############################ BINNING STRATEGIES ############################
def scott_rule(x):
    n = len(x)
    sigma = np.std(x)
    bin_width = 3.5 * sigma / (n ** (1/3))
    
    # Handle zero bin_width (constant data or very low variance)
    if bin_width == 0 or np.max(x) == np.min(x):
        return max(2, min(int(np.sqrt(n)), n//5))  # Fall back to sqrt rule
    
    bins = int((np.max(x) - np.min(x)) / bin_width)
    return max(2, min(bins, n//5))  # Reasonable bounds

def knuth_rule(x):
    """Knuth's rule for optimal binning."""
    n = len(x)
    
    def cost_function(k):
        if k < 2:
            return np.inf
        
        hist, _ = np.histogram(x, bins=int(k))
        hist = hist[hist > 0]  # Remove empty bins
        
        if len(hist) < 2:
            return np.inf
        
        # Log-likelihood
        log_likelihood = np.sum(hist * np.log(hist / n))
        
        # BIC penalty
        penalty = 0.5 * (k - 1) * np.log(n)
        
        return -log_likelihood + penalty
    
    # Search for optimal k
    k_range = range(2, min(50, n//5))
    costs = [cost_function(k) for k in k_range]
    return k_range[np.argmin(costs)]

def freedman_diaconis_rule(x):
    n = len(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    bin_width = 2 * iqr / (n ** (1/3))
    
    # Handle zero bin_width (no variability in data)
    if bin_width == 0 or np.max(x) == np.min(x):
        return max(2, min(int(np.sqrt(n)), n//5))  # Fall back to sqrt rule
    
    bins = int((np.max(x) - np.min(x)) / bin_width)
    return max(2, min(bins, n//5))

class OptimalMIEstimator:
    def __init__(self, method='auto'):
        self.method = method.lower()
        self._methods = ['fd', 'scott', 'knuth', 'sturges', 'sqrt']

    def _bins_from_method(self, x, method):
        '''
        Determine the number of bins for a given method.
        '''
        if method == 'fd':
            return freedman_diaconis_rule(x)
        if method == 'scott':
            return scott_rule(x)
        if method == 'knuth':
            return knuth_rule(x)
        if method == 'sturges':
            return max(1, int(np.ceil(np.log2(len(x))) + 1))
        if method == 'sqrt':
            return max(1, int(np.ceil(np.sqrt(len(x)))))
        raise ValueError(f"Unknown binning method: {method}")

    def _digitize(self, x, bins):
        '''
        Digitize x into bins, ensuring at least 2 unique edges.
        '''
        if bins <= 1 or np.allclose(x, x[0]):
            return np.zeros_like(x, dtype=int)
        edges = np.histogram_bin_edges(x, bins=bins)
        edges = np.unique(edges)
        if edges.size <= 2:
            return np.zeros_like(x, dtype=int)
        # Use interior edges for digitization, handling edge case
        interior_edges = edges[1:-1] if edges.size > 2 else []
        if len(interior_edges) == 0:
            return np.zeros_like(x, dtype=int)
        return np.digitize(x, interior_edges, right=False)

    def estimate(self, x, y):
        '''
        Estimate the mutual information between x and y using optimal binning.
        '''
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if x.size == 0:
            return 0.0, 1, self.method

        candidates = [self.method] if self.method != 'auto' else self._methods
        best_mi, best_bins, best_method = -np.inf, None, None

        for method in candidates:
            try:
                bins = self._bins_from_method(x, method)
                bins = max(1, bins)
                x_disc = self._digitize(x, bins)
                
                # Use actual number of unique values as bins for MI calculation
                actual_bins = len(np.unique(x_disc))
                mi = mutual_information_from_scratch(y, x_disc, bins=actual_bins)
                
                if mi > best_mi:
                    best_mi, best_bins, best_method = mi, bins, method
            except Exception:
                continue

        if best_method is None:
            # More robust fallback
            best_bins = max(2, min(int(np.ceil(np.sqrt(len(x)))), len(np.unique(x))))
            x_disc = self._digitize(x, best_bins)
            
            # Final validation for fallback
            if len(np.unique(x_disc)) <= 1:
                best_mi = 0.0  # No information if all values are the same
            else:
                actual_bins = len(np.unique(x_disc))
                best_mi = mutual_information_from_scratch(y, x_disc, bins=actual_bins)
            best_method = 'sqrt'

        return float(best_mi), int(best_bins), best_method


############################ OUTLIER DETECTION STRATEGIES ############################

def robust_outlier_detection(x):
    '''
    Robust outlier detection using the IQR method.
    '''
    if isinstance(x, pd.Series):
        values = x.astype(float)
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            std = values.std()
            if std == 0:
                return pd.Series(False, index=x.index)
            z = (values - values.mean()).abs() / std
            return z > 3
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (values < lower) | (values > upper)

    arr = np.asarray(x, dtype=float)
    q1 = np.nanpercentile(arr, 25)
    q3 = np.nanpercentile(arr, 75)
    iqr = q3 - q1
    if iqr == 0:
        std = np.nanstd(arr)
        if std == 0:
            return np.zeros_like(arr, dtype=bool)
        z = np.abs(arr - np.nanmean(arr)) / std
        return z > 3
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (arr < lower) | (arr > upper)