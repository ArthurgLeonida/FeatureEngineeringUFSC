from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mutual_info_score
import numpy as np
import pandas as pd

############################ DEPENDENCE MEASURES ############################

def mutual_information(x, y, bins=10):
    """Calculates mutual information from scratch using NumPy."""
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    
    # Discretize x if it is continuous/numeric
    if np.issubdtype(x.dtype, np.number) and len(np.unique(x)) > bins:
        hist, bin_edges = np.histogram(x, bins=bins)
        x_discrete = np.digitize(x, bin_edges[:-1]) - 1
    else:
        # If already categorical, map to integer indices
        _, x_discrete = np.unique(x, return_inverse=True)
        
    # Handle y (target) - map to integer indices
    _, y_discrete = np.unique(y, return_inverse=True)
    
    # Calculate Joint Probability Distribution P(X, Y)
    n_x = len(np.unique(x_discrete))
    n_y = len(np.unique(y_discrete))
    
    # bins argument in histogram2d takes the number of edges, so we pass max index + 1
    joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=[n_x, n_y])
    p_xy = joint_hist / n
    
    # Calculate Marginal Distributions P(X) and P(Y)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # Calculate MI: sum( p(x,y) * log( p(x,y) / (p(x)*p(y)) ) )
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 0: # Avoid log(0)
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
                
    return mi

def hsic(X, Y, sigma_X=1.0, sigma_Y=1.0):
    """Calculates HSIC using manual kernel implementation."""
    n = len(X)
    if n < 2: return 0.0
    
    # Use manual kernel instead of sklearn
    K = manual_rbf_kernel(X, sigma=sigma_X)
    L = manual_rbf_kernel(Y, sigma=sigma_Y)
    
    H = np.eye(n) - (1/n) * np.ones((n,n))
    Kc = H @ K @ H
    Lc = H @ L @ H
    
    return np.trace(Kc @ Lc) / ((n - 1)**2)

def manual_rbf_kernel(X, sigma=1.0):
    """Manual implementation of RBF Kernel."""
    X = np.asarray(X).reshape(-1, 1)
    # Compute squared Euclidean distances
    # (x - y)^2 = x^2 + y^2 - 2xy
    X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
    dists_sq = X_sq + X_sq.T - 2 * np.dot(X, X.T)
    gamma = 1.0 / (2 * sigma**2)
    return np.exp(-gamma * dists_sq)

############################ BINNING STRATEGIES ############################

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

def adaptive_binning(x):
    """Chooses a binning strategy based on data characteristics."""
    x = x.dropna()
    n = len(x)
    x_unique = x.nunique()
    
    # Logic is the same, but only for x
    if x_unique <= 10:
        bins_x = x_unique
    elif n < 50:
        bins_x = int(np.sqrt(n))
    elif n < 200:
        bins_x = freedman_diaconis_rule(x)
    else:
        bins_x = knuth_rule(x)
    
    # Return only the bins for x
    return max(2, min(bins_x, 50))

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
    consensus_outliers = z_outliers & iqr_outliers
    return consensus_outliers