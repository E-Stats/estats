import numpy as np
from scipy.stats import zscore, norm
from scipy.special import hyp1f1, gamma
# from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist

def normality_statistic(x: np.array) -> np.float64:
    """
    Returns the statistic for the univariate normality test
    """
    x = np.array(x)
    n = len(x)

    y = np.sort(zscore(x, ddof=1)) 
    k = np.arange(1 - n, n, 2)
    
    statistic = 2 * (np.sum(2 * y * norm.cdf(y) + 2 * norm.pdf(y)) - n/np.sqrt(np.pi) - np.mean(k * y))
    
    return statistic

# def m_normality_statistic(x: np.array) -> np.float64:
#     """
#     Returns the statistic for the multivariate normality test
#     """
#     n = len(x)
#     y = zscore(x)
#     d = x.shape[1]
#     neg_norms = - (np.linalg.norm(y, axis=1)**2 / 2)
#     k = gamma((d+1) / 2) / gamma(d/2)

#     first_term = np.sqrt(2) * k * (2/n) * np.sum(hyp1f1(-1/2, d/2, neg_norms))
#     second_term = 2 * k +  np.sum(distance_matrix(y, y)) / n**2

#     return n * (first_term - second_term)

def m_normality_statistic(x: np.array) -> float:
    """
    Computes the E-statistic for the multivariate normality test.

    Parameters:
        x (np.array): Input data array of shape (n_samples, n_features).
    Returns:
        float: E-statistic for the multivariate normality test.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]

    n, d = x.shape
    if n < 2:
        raise ValueError("Sample size must be at least 2")

    # Step 1: Center and whiten the data
    x_mean = np.mean(x, axis=0)
    z = x - x_mean  # Center the data
    cov_matrix = np.cov(z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    whitening_matrix = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
    y = z @ whitening_matrix

    if not np.all(np.isfinite(y)):
        raise ValueError("Non-finite values encountered during whitening")

    # Step 2: Constants for the E-statistic
    const = np.exp(np.log(gamma((d + 1) / 2)) - np.log(gamma(d / 2)))
    mean2 = 2 * const

    # Step 3: Compute squared norms and hypergeometric term
    ysq = np.sum(y ** 2, axis=1)
    mean1 = np.sqrt(2) * const * np.mean(hyp1f1(-0.5, d / 2, -ysq / 2))

    # Step 4: Pairwise distances
    pairwise_dists = pdist(y)
    mean3 = 2 * np.sum(pairwise_dists) / (n ** 2)

    # Step 5: E-statistic
    e_statistic = n * (2 * mean1 - mean2 - mean3)
    return e_statistic
