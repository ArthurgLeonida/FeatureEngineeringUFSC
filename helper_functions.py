from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mutual_info_score
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity
from typing import Optional, Union
from collections.abc import Iterable
import warnings
warnings.filterwarnings('ignore')

############################ DEPENDENCE MEASURES ############################

def mutual_information(x, y, bins=10):
    """Calculates mutual information using the standard sklearn method."""
    # Discretize continuous variable x before calculating MI
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
        x_binned = pd.cut(x, bins=bins, labels=False, duplicates='drop')
    else:
        x_binned = x
        
    # Use the direct function from scikit-learn
    return mutual_info_score(y, x_binned)

def hsic(X, Y, sigma_X=1.0, sigma_Y=1.0):
    """Calculates the Hilbert-Schmidt Independence Criterion."""
    n = len(X)
    if n < 2: return 0.0
    X, Y = np.asarray(X).reshape(-1, 1), np.asarray(Y).reshape(-1, 1)
    K, L = rbf_kernel(X, gamma=1/(2*sigma_X**2)), rbf_kernel(Y, gamma=1/(2*sigma_Y**2))
    H = np.eye(n) - (1/n) * np.ones((n,n))
    Kc, Lc = H @ K @ H, H @ L @ H
    return np.trace(Kc @ Lc) / ((n - 1)**2)

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

ArrayLike = Union[np.ndarray, Iterable[float]]


def find_optimal_bins(data, max_bins=100, plot=True, verbose=True):
    """
    Find optimal number of bins using multiple state-of-the-art methods.
    
    Parameters:
    - data: array-like, the dataset
    - max_bins: int, maximum number of bins to test
    - plot: bool, whether to show plots
    - verbose: bool, whether to print results
    
    Returns:
    - dict with optimal bins for each method
    """
    
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]  # Remove NaN values
    n = len(data)
    
    if n == 0:
        raise ValueError("Data contains no valid values")
    
    results = {}
    
    # 1. Sturges' Rule (classic)
    sturges = int(np.ceil(np.log2(n) + 1))
    results['Sturges'] = sturges
    
    # 2. Scott's Rule (based on standard deviation)
    h_scott = 3.5 * np.std(data) / (n ** (1/3))
    scott = int(np.ceil((np.max(data) - np.min(data)) / h_scott))
    results['Scott'] = max(1, min(scott, max_bins))
    
    # 3. Freedman-Diaconis Rule (robust to outliers)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr > 0:
        h_fd = 2 * iqr / (n ** (1/3))
        fd = int(np.ceil((np.max(data) - np.min(data)) / h_fd))
        results['Freedman-Diaconis'] = max(1, min(fd, max_bins))
    else:
        results['Freedman-Diaconis'] = sturges
    
    # 4. Square Root Rule
    sqrt_rule = int(np.ceil(np.sqrt(n)))
    results['Square Root'] = sqrt_rule
    
    # 5. Doane's Rule (extension of Sturges for skewed data)
    g1 = stats.skew(data)
    sigma_g1 = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
    doane = int(np.ceil(np.log2(n) + 1 + np.log2(1 + abs(g1) / sigma_g1)))
    results['Doane'] = doane
    
    # 6. Cross-Validation Method (computationally intensive but robust)
    bin_range = range(max(2, int(n**0.3)), min(max_bins, int(n**0.6)))
    cv_scores = []
    
    for k in bin_range:
        try:
            # Create histogram and use it as basis for KDE evaluation
            hist, edges = np.histogram(data, bins=k, density=True)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            
            # Weighted score based on histogram smoothness and data fit
            # Penalize both underfitting (too few bins) and overfitting (too many bins)
            smoothness = -np.sum(np.diff(hist)**2)  # Prefer smoother histograms
            coverage = np.sum(hist > 0) / len(hist)  # Prefer good data coverage
            
            # Combine metrics with penalty for extreme bin counts
            penalty = abs(k - np.sqrt(n)) / np.sqrt(n)  # Penalize deviation from sqrt(n)
            score = smoothness + coverage - penalty
            cv_scores.append(score)
            
        except:
            cv_scores.append(-np.inf)
    
    if cv_scores:
        cv_optimal = bin_range[np.argmax(cv_scores)]
        results['Cross-Validation'] = cv_optimal
    else:
        results['Cross-Validation'] = sturges
    
    # 7. Fixed BIC approach using proper multinomial likelihood
    def compute_bic_optimal(data, max_bins_test):
        """
        Compute BIC for histogram model selection using proper likelihood.
        """
        N = len(data)
        min_bins_test = max(2, int(np.log2(N)))  # Start from reasonable minimum
        
        bic_scores = []
        
        for k in range(min_bins_test, max_bins_test + 1):
            try:
                # Create histogram
                hist, _ = np.histogram(data, bins=k, density=False)
                
                # Log-likelihood for multinomial distribution
                # Remove empty bins to avoid log(0)
                observed = hist[hist > 0]
                
                if len(observed) >= 2:
                    # Log-likelihood: sum of n_i * log(p_i) where p_i = n_i/N
                    log_likelihood = np.sum(observed * np.log(observed / N))
                    
                    # BIC = -2 * log_likelihood + (k-1) * log(N)
                    # k-1 parameters for multinomial (probabilities sum to 1)
                    bic = -2 * log_likelihood + (k - 1) * np.log(N)
                    bic_scores.append(bic)
                else:
                    bic_scores.append(np.inf)
                    
            except (ValueError, RuntimeWarning):
                bic_scores.append(np.inf)
        
        if bic_scores and not all(np.isinf(bic_scores)):
            bic_optimal_idx = np.argmin(bic_scores)
            bic_optimal = bic_optimal_idx + min_bins_test
            return max(2, min(bic_optimal, max_bins))
        else:
            return sturges
    
    # Compute BIC with reasonable range
    max_bins_bic = min(max_bins, int(2 * np.sqrt(n)))
    results['BIC'] = compute_bic_optimal(data, max_bins_bic)
    
    # 8. Fixed Bayesian Blocks implementation
    def _fitness_args_of(func):
        return func.__code__.co_varnames[: func.__code__.co_argcount]

    class FitnessFunc:
        def __init__(self, p0: float = 0.05, gamma: Optional[float] = None, ncp_prior: Optional[float] = None) -> None:
            self.p0 = p0
            self.gamma = gamma
            self.ncp_prior = ncp_prior

        def validate_input(self, t: ArrayLike, x: Optional[ArrayLike] = None, sigma: Optional[Union[ArrayLike, float]] = None):
            t = np.asarray(t, dtype=float)
            if t.ndim != 1 or t.size == 0:
                raise ValueError("t must be non-empty 1D array")
            unq_t, unq_ind, unq_inv = np.unique(t, return_index=True, return_inverse=True)

            if x is None:
                if sigma is not None:
                    raise ValueError("If sigma is provided, x must be provided for 'measures'.")
                sigma = 1.0
                if unq_t.size == t.size:
                    x = np.ones_like(unq_t)
                else:
                    x = np.bincount(unq_inv).astype(float)
                t = unq_t
            else:
                x = np.asarray(x, dtype=float)
                if x.shape not in [(), (1,), (t.size,)]:
                    raise ValueError("x shape must be scalar or match t")
                if unq_t.size != t.size:
                    raise ValueError("Repeated values in t not supported when x is provided")
                t = unq_t
                x = x[unq_ind] if x.shape == (t.size,) else np.full_like(t, float(x))

            if sigma is None:
                sigma = 1.0
            sigma = np.asarray(sigma, dtype=float)
            if sigma.shape not in [(), (1,), (t.size,)]:
                raise ValueError("sigma shape must be scalar or match t/x")
            if sigma.shape != (t.size,):
                sigma = np.full_like(t, float(sigma))
            return t, x, sigma

        def p0_prior(self, N: int) -> float:
            return 4.0 - np.log(73.53 * self.p0 * (N ** -0.478))

        def compute_ncp_prior(self, N: int) -> float:
            if self.ncp_prior is not None:
                return float(self.ncp_prior)
            if self.gamma is not None:
                return -np.log(float(self.gamma))
            if self.p0 is not None:
                return float(self.p0_prior(N))
            raise ValueError("Cannot compute ncp_prior: set p0 or gamma or ncp_prior.")

        @property
        def _fitness_args(self):
            return _fitness_args_of(self.fitness)

        def fitness(self, **kwargs) -> np.ndarray:
            raise NotImplementedError

        def fit(self, t: ArrayLike, x: Optional[ArrayLike] = None, sigma: Optional[Union[ArrayLike, float]] = None) -> np.ndarray:
            t, x, sigma = self.validate_input(t, x, sigma)

            if "a_k" in self._fitness_args:
                a_raw = 1.0 / (sigma * sigma)
            if "b_k" in self._fitness_args:
                b_raw = x / (sigma * sigma)
            if "c_k" in self._fitness_args:
                c_raw = (x * x) / (sigma * sigma)

            edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
            block_len_from_right = t[-1] - edges

            N = t.size
            best = np.zeros(N, dtype=float)
            last = np.zeros(N, dtype=int)
            ncp_prior = self.compute_ncp_prior(N)

            for R in range(N):
                kw = {}
                if "T_k" in self._fitness_args:
                    kw["T_k"] = block_len_from_right[: R + 1] - block_len_from_right[R + 1]
                if "N_k" in self._fitness_args:
                    kw["N_k"] = np.cumsum(x[: R + 1][::-1])[::-1]
                if "a_k" in self._fitness_args:
                    kw["a_k"] = 0.5 * np.cumsum(a_raw[: R + 1][::-1])[::-1]
                if "b_k" in self._fitness_args:
                    kw["b_k"] = -np.cumsum(b_raw[: R + 1][::-1])[::-1]
                if "c_k" in self._fitness_args:
                    kw["c_k"] = 0.5 * np.cumsum(c_raw[: R + 1][::-1])[::-1]

                fit_vec = self.fitness(**kw)
                A_R = fit_vec - ncp_prior
                if R > 0:
                    A_R[1:] += best[:R]

                i_max = int(np.argmax(A_R))
                last[R] = i_max
                best[R] = A_R[i_max]

            cps = []
            ind = N
            while ind > 0:
                cps.append(ind)
                ind = last[ind - 1]
                if ind == 0:
                    cps.append(0)
                    break
            cps = np.array(cps[::-1], dtype=int)
            return edges[cps]

    class Events(FitnessFunc):
        def fitness(self, N_k: np.ndarray, T_k: np.ndarray) -> np.ndarray:
            out = np.zeros_like(N_k, dtype=float)
            mask = N_k > 0
            rate = np.divide(N_k, T_k, out=np.zeros_like(N_k, dtype=float), where=mask)
            ln_rate = np.zeros_like(N_k, dtype=float)
            np.log(rate, out=ln_rate, where=mask)
            np.multiply(N_k, ln_rate, out=out, where=mask)
            return out

        def validate_input(self, t, x, sigma):
            t, x, sigma = super().validate_input(t, x, sigma)
            if x is not None and (np.any(x < 0) or np.any(x % 1 != 0)):
                raise ValueError("For 'events', x must be non-negative integer counts.")
            return t, x, sigma

    class RegularEvents(FitnessFunc):
        def __init__(self, dt: float, p0: float = 0.05, gamma: Optional[float] = None, ncp_prior: Optional[float] = None) -> None:
            if dt <= 0:
                raise ValueError("dt must be positive for 'regular_events'.")
            self.dt = float(dt)
            super().__init__(p0=p0, gamma=gamma, ncp_prior=ncp_prior)

        def validate_input(self, t, x=None, sigma=None):
            t, x, sigma = super().validate_input(t, x, sigma)
            if x is not None and not np.all((x == 0) | (x == 1)):
                raise ValueError("'regular_events' requires x in {0,1}.")
            return t, x, sigma

        def fitness(self, T_k: np.ndarray, N_k: np.ndarray) -> np.ndarray:
            M_k = T_k / self.dt
            eps = 1e-12
            q = np.clip(N_k / np.maximum(M_k, eps), eps, 1 - eps)
            return N_k * np.log(q) + (M_k - N_k) * np.log(1 - q)

    class PointMeasures(FitnessFunc):
        def fitness(self, a_k: np.ndarray, b_k: np.ndarray) -> np.ndarray:
            return (b_k * b_k) / (4.0 * a_k)

        def validate_input(self, t, x, sigma):
            if x is None:
                raise ValueError("'measures' requires x (and optionally sigma).")
            return super().validate_input(t, x, sigma)

    def bayesian_blocks(t: ArrayLike, x: Optional[ArrayLike] = None, sigma: Optional[Union[ArrayLike, float]] = None,
                        fitness: Union[str, FitnessFunc] = "events", **kwargs) -> np.ndarray:
        FITNESS = {"events": Events, "regular_events": RegularEvents, "measures": PointMeasures}
        fitcls_or_obj = FITNESS.get(fitness, fitness)
        if isinstance(fitcls_or_obj, type) and issubclass(fitcls_or_obj, FitnessFunc):
            fitfunc = fitcls_or_obj(**kwargs)
        elif isinstance(fitcls_or_obj, FitnessFunc):
            fitfunc = fitcls_or_obj
        else:
            raise ValueError("fitness must be 'events', 'regular_events', 'measures', a FitnessFunc subclass, or instance.")
        return fitfunc.fit(t, x, sigma)
    
    # Fixed Bayesian Blocks computation - convert edges to number of bins
    def compute_bb_bins(data, fitness_type, **kwargs):
        """Helper function to compute BB bins and convert to integer bin count"""
        try:
            edges = bayesian_blocks(data, fitness=fitness_type, **kwargs)
            # Convert edges to number of bins
            num_bins = len(edges) - 1
            return max(1, min(num_bins, max_bins))
        except Exception as e:
            if verbose:
                print(f"Warning: {fitness_type} Bayesian Blocks failed: {e}")
            return sturges
    
    # Compute Bayesian Blocks variants with proper error handling
    results['BB Events'] = compute_bb_bins(data, 'events', p0=0.05)
    
    # For regular events, create binary data (0/1) based on threshold
    # Create binary data for regular events (above/below median)
    binary_data = (data > np.median(data)).astype(int)
    data_range = np.max(data) - np.min(data)
    dt = data_range / len(data) if len(data) > 1 else 1.0
    results['BB Regular'] = compute_bb_bins(data, 'regular_events', dt=dt, p0=0.05)

    # For measures, use data with its mean and std
    # Use sorted data for measures
    sorted_data = np.sort(data)
    x_measures = np.full_like(sorted_data, np.mean(data))
    sigma_measures = np.std(data)
    if sigma_measures == 0:
        sigma_measures = 1.0
    results['BB Measures'] = compute_bb_bins(sorted_data, 'measures', 
                                            x=x_measures, sigma=sigma_measures, p0=0.05)

    
    # Calculate ensemble recommendation (weighted average of methods)
    weights = {
        'Freedman-Diaconis': 0.2,   # Robust to outliers
        'BIC': 0.15,                # Model selection based
        'Cross-Validation': 0.1,    # Data-driven
        'BB Events': 0.08,          # Events-based BB
        'Scott': 0.07,              # Standard deviation based
        'BB Measures': 0.05,        # Point measures BB
        'Doane': 0.04,              # Accounts for skewness
        'BB Regular': 0.03,         # Regular events BB
        'Square Root': 0.02,        # Simple baseline
        'Sturges': 0.01             # Historical baseline
    }
    
    ensemble = int(np.round(sum(results[method] * weights[method] 
                                for method in weights.keys())))
    results['Ensemble'] = max(1, min(ensemble, max_bins))
    
    if verbose:
        print("Optimal Number of Bins by Different Methods:")
        print("-" * 45)
        for method, bins in results.items():
            print(f"{method:20}: {bins:3d} bins")
        print("-" * 45)
        print(f"Recommended (Ensemble): {results['Ensemble']} bins")
    
    if plot:
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        methods_to_plot = ['Sturges', 'Scott', 'Freedman-Diaconis', 
                          'BB Regular', 'Cross-Validation', 'Ensemble']
        
        for i, method in enumerate(methods_to_plot):
            if i < len(axes) and method in results:
                axes[i].hist(data, bins=results[method], alpha=0.7, 
                           color=plt.cm.Set3(i), edgecolor='black', linewidth=0.5)
                axes[i].set_title(f'{method}\n({results[method]} bins)')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Histogram Comparison: Different Bin Selection Methods', 
                    fontsize=16, y=1.02)
        plt.show()
        
        # Show method comparison
        methods = list(results.keys())
        bins = list(results.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(methods, bins, color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Bins')
        plt.title('Optimal Number of Bins by Different Methods')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, bins):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    return results