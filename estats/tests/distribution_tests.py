import numpy as np
from scipy.stats import zscore, norm
from scipy.special import hyp1f1, gamma
from scipy.spatial import distance_matrix

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

def m_normality_statistic(x: np.array) -> np.float64:
    """
    Returns the statistic for the multivariate normality test
    """
    n = len(x)
    y = zscore(x)
    d = x.shape[1]
    neg_norms = - (np.linalg.norm(y, axis=1)**2 / 2)
    k = gamma((d+1) / 2) / gamma(d/2)

    first_term = np.sqrt(2) * k * (2/n) * np.sum(hyp1f1(-1/2, d/2, neg_norms))
    second_term = 2 * k +  np.sum(distance_matrix(y, y)) / n**2

    return n * (first_term - second_term)

    

def normality_etest(x: np.array, iterations: int = 1000, statistic = normality_statistic) -> dict:
    """
    Given a data matrix x which is n * m, tests whether the  m random variable are normal and returns the test statistic and p-value in a dictionary
    """
    statistic = normality_statistic
    if np.ndim(x) > 1:
        statistic = m_normality_statistic
    observed_statistic = statistic(x)
    simulations = []

    for i in range(iterations):
        sim_x = norm.rvs(size = x.shape)
        simulations.append(statistic(sim_x))

    p_value = np.mean(simulations > observed_statistic) 
    return {'observed_statistic': observed_statistic, 'p_value': p_value}
