from cor import dcor
import numpy as np
from scipy.stats import zscore, norm, cauchy, kstest, t
from sklearn import datasets

iris = datasets.load_iris()['data']


def normality_statistic(x: np.array) -> np.float64:
    """
    Returns the statistic for the univariate normality test
    """
    x = np.array(x)
    n = len(x)
    s = np.std(x, ddof=1) 
        
    y = np.sort((x - np.mean(x)) / s) 
    k = np.arange(1 - n, n, 2)
    
    statistic = 2 * (np.sum(2 * y * norm.cdf(y) + 2 * norm.pdf(y)) - 
                  n/np.sqrt(np.pi) - 
                  np.mean(k * y))
    
    return statistic
    

def normality_etest(x: np.array, permutations: int = 1000) -> dict:
    """
    Given a vector x, tests whether that vector is normal and returns the test statistic and p-value in a dictionary
    """
    observed_statistic = normality_statistic(x)
    simulations = []
    n = len(x)

    for i in range(permutations):
        sim_x = norm.rvs(size = n)
        simulations.append(normality_statistic(sim_x))

    p_value = np.mean(simulations > observed_statistic) 
    return {'observed_statistic': observed_statistic, 'p_value': p_value}



x = t.rvs(5, size = 1000)
print(f"etsest: {normality_etest(x)}")
print(f'kolmogrov test: {kstest(x, cdf=norm.cdf)}')

