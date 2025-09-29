from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mutual_info_score
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

############################ DEPENDENCE MEASURES ############################

def mutual_information_from_scratch(x, y, bins=10):
    """
    Implement Mutual Information using histogram method
    MI(X,Y) = H(X) + H(Y) - H(X,Y)
    """
    def entropy(data, bins):
        """Calculate entropy of a variable"""
        hist, _ = np.histogram(data, bins=bins)
        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]
        # Convert to probabilities
        probs = hist / hist.sum()
        # Calculate entropy: -sum(p * log(p))
        return -np.sum(probs * np.log2(probs))
    
    def joint_entropy(x, y, bins):
        """Calculate joint entropy H(X,Y)"""
        # Create 2D histogram
        hist, _, _ = np.histogram2d(x, y, bins=bins)
        # Remove zeros
        hist = hist[hist > 0]
        # Convert to probabilities
        probs = hist / hist.sum()
        # Calculate joint entropy
        return -np.sum(probs * np.log2(probs))
    
    # Calculate individual entropies
    h_x = entropy(x, bins)
    h_y = entropy(y, bins)
    
    # Calculate joint entropy
    h_xy = joint_entropy(x, y, bins)
    
    # Mutual Information = H(X) + H(Y) - H(X,Y)
    mi = h_x + h_y - h_xy
    
    return mi

def rbf_kernel(x, y, sigma=None):
    """
    Radial Basis Function (Gaussian) kernel
    k(x,y) = exp(-||x-y||²/(2σ²))
    """
    if sigma is None:
        # Median heuristic: use median distance as bandwidth
        distances = pdist(x.reshape(-1, 1))
        sigma = np.median(distances)
        if sigma == 0:
            sigma = 1.0
        
    # Calculate pairwise distances correctly
    x_expanded = x.reshape(-1, 1)
    # Create distance matrix: ||x_i - x_j||²
    distances_sq = np.sum((x_expanded[:, None] - x_expanded[None, :])**2, axis=2)
    
    # Apply Gaussian kernel
    kernel_matrix = np.exp(-distances_sq / (2 * sigma**2))
    return kernel_matrix

def center_kernel_matrix(K):
    """
    Center the kernel matrix (remove mean similarities)
    H @ K @ H where H = I - (1/n) * ones_matrix
    """
    n = K.shape[0]
    # Create centering matrix H = I - (1/n) * ones_matrix
    ones = np.ones((n, n)) / n
    H = np.eye(n) - ones
    
    # Center the kernel: H @ K @ H
    K_centered = H @ K @ H
    return K_centered

def hsic_from_scratch(x, y, sigma_x=None, sigma_y=None):
    """
    Implement HSIC (Hilbert-Schmidt Independence Criterion)
    Measures dependence using kernel similarities
    """
    n = len(x)
    
    # Create kernel matrices
    K_x = rbf_kernel(x, x, sigma_x)
    K_y = rbf_kernel(y, y, sigma_y)
    
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
    bins = int((np.max(x) - np.min(x)) / bin_width)
    return max(2, min(bins, n//5))

class OptimalMIEstimator:
    def __init__(self, method='auto'):
        self.method = method
        self.binning_methods = {
            'sqrt': lambda x, y: int(np.sqrt(len(x))),
            'scott': lambda x, y: scott_rule(x),
            'freedman_diaconis': lambda x, y: freedman_diaconis_rule(x),
            'knuth': lambda x, y: knuth_rule(x)
        }
    
    def estimate(self, x, y):
        """Estimate MI with optimal binning"""
        # Clean data
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        # Choose binning method
        if self.method == 'auto':
            method = self._choose_method(x, y)
        else:
            method = self.method
        
        # Get optimal bins
        bins = self.binning_methods[method](x, y)
        
        # Estimate MI
        mi = self._compute_mi_histogram(x, y, bins)
        
        return mi, bins, method
    
    def _choose_method(self, x, y):
        """Automatically choose best method based on data"""
        n = len(x)
        
        if n < 50:
            return 'sqrt'
        elif n < 200:
            skewness = (np.mean((x - np.mean(x))**3)) / (np.std(x)**3)
            if abs(skewness) > 1:
                return 'scott'
            else:
                return 'freedman_diaconis'
        else:
            return 'knuth'
    
    def _compute_mi_histogram(self, x, y, bins):
        """Compute MI using histogram method"""
        return mutual_information_from_scratch(x, y, bins=bins)

############################ OUTLIER DETECTION STRATEGIES ############################

def robust_outlier_detection(x):
    """Detects outliers using multiple methods and returns a consensus."""
    # Method 1: Z-score (assumes normality)
    z_outliers = abs((x - x.mean()) / x.std()) > 3
    
    # Method 2: IQR method (non-parametric)
    Q1, Q3 = x.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    iqr_outliers = (x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR))
    
    # Method 3: Percentile method (robust to distribution shape)
    p5, p95 = x.quantile([0.05, 0.95])
    percentile_outliers = (x < p5) | (x > p95)
    
    # Consensus approach: flagged by multiple methods
    consensus_outliers = z_outliers & iqr_outliers & percentile_outliers
    return consensus_outliers