import numpy as np
from estats.tests.distribution_tests import m_normality_statistic

def test_mvnorm_statistic():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Test Case 1: Multivariate normal data
    data_normal = np.random.normal(loc=0, scale=1, size=(100, 5))  # 100 samples, 5 dimensions
    statistic_normal = m_normality_statistic(data_normal)
    print("Statistic for normal data (100x5):", statistic_normal)
    
    # Test Case 2: Non-normal data (uniform distribution)
    data_non_normal = np.random.uniform(low=-5, high=5, size=(100, 5))
    statistic_non_normal = m_normality_statistic(data_non_normal)
    print("Statistic for uniform data (100x5):", statistic_non_normal)
    
    # Test Case 3: Larger multivariate normal data
    data_large_normal = np.random.normal(loc=0, scale=1, size=(500, 10))  # 500 samples, 10 dimensions
    statistic_large_normal = m_normality_statistic(data_large_normal)
    print("Statistic for normal data (500x10):", statistic_large_normal)
    
    # Test Case 4: Larger non-normal data (uniform distribution)
    data_large_non_normal = np.random.uniform(low=-10, high=10, size=(500, 10))
    statistic_large_non_normal = m_normality_statistic(data_large_non_normal)
    print("Statistic for uniform data (500x10):", statistic_large_non_normal)

# Run the test
test_mvnorm_statistic()
