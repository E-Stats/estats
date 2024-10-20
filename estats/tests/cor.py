import numpy as np
from scipy.spatial import distance_matrix

def double_center_matrix(m: np.array) -> np.array:
    """
    Given a matrix m, returns the doubly centered matrix k, such that,
    k_ij = m - mean(the ith row of m) - mean(the jth column of m) + total mean.

    """
    row_means = np.mean(m, 1)
    column_means = np.mean(m, 0)
    total_mean = np.mean(row_means)
    
    return m - row_means - column_means + total_mean


def dcov(x: np.array, y : np.array) -> np.float64:
    """
    Returns the distance covariance, which is an always positive number representing the degree of dependency of two random variables. 
    """
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)

    a = distance_matrix(x, x)
    b = distance_matrix(y, y)
    a_centered, b_centered = double_center_matrix(a), double_center_matrix(b)

    return np.sqrt(np.mean(a_centered * b_centered))

def dcor(x: np.array, y: np.array) -> np.float64:
    """
    Returns an estimation of the distance correlation, R, which has  the following two properties:
    1- 0 <= R <= 1.
    2- R = 0 iff x and y are independent.
    """
    var_x = dcov(x, x)
    var_y = dcov(y, y)

    if np.isclose(var_x * var_y, 0):
        return 0
    
    return dcov(x, y) / np.sqrt(var_x * var_y)


