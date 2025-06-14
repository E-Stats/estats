import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def compute_distance_matrix(data, metric='euclidean'):
    """Compute the pairwise distance matrix."""
    pairwise_dist = pdist(data, metric=metric)
    return squareform(pairwise_dist)


def re_center_distance_matrix(D):
    """Re-center the distance matrix."""
    row_mean = np.mean(D, axis=1, keepdims=True)
    col_mean = np.mean(D, axis=0, keepdims=True)
    total_mean = np.mean(D)
    return D - row_mean - col_mean + total_mean


def distance_covariance(R_X, R_Y):
    """Compute the distance covariance."""
    return np.sum(R_X * R_Y) / (R_X.shape[0] ** 2)


def projected_gradient_descent(X, Y, R_Y, learning_rate=0.01, max_iter=500, tol=1e-6):
    """Optimize projection vector using projected gradient descent."""
    n, p = X.shape
    u = np.random.randn(p)
    u /= np.linalg.norm(u)  # Normalize

    for _ in range(max_iter):
        # Gradient of the objective
        grad = np.zeros_like(u)
        for i in range(n):
            for j in range(n):
                diff = X[i] - X[j]
                grad += R_Y[i, j] * (np.sign(np.dot(u, diff)) * diff)

        grad /= -(n ** 2)

        # Update and project back to unit sphere
        u_new = u - learning_rate * grad
        u_new /= np.linalg.norm(u_new)

        # Check convergence
        if np.linalg.norm(u_new - u) < tol:
            break
        u = u_new

    return u


def optimize_projections(X, Y, num_dims=1):
    """Find multiple orthogonal projection vectors."""
    n, p = X.shape
    R_Y = re_center_distance_matrix(compute_distance_matrix(Y))
    projections = []
    orthogonal_X = X.copy()

    for _ in range(num_dims):
        u = projected_gradient_descent(orthogonal_X, Y, R_Y)
        projections.append(u)
        # Deflate the data for orthogonality
        projection_matrix = np.outer(u, u)
        orthogonal_X -= orthogonal_X @ projection_matrix

    return np.array(projections)


# Generate shared latent variable
np.random.seed(42)
Z = np.random.randn(75, 5)

# Create X and Y from Z with different transformations
X = Z @ np.random.randn(5, 10) + 0.1 * np.random.randn(75, 10)
Y = Z @ np.random.randn(5, 8) + 0.1 * np.random.randn(75, 8)

# Compute distance matrices
D_X = compute_distance_matrix(X)
D_Y = compute_distance_matrix(Y)

# Re-center distance matrices
R_X = re_center_distance_matrix(D_X)
R_Y = re_center_distance_matrix(D_Y)

# Optimize projections
num_dimensions = 2
projections = optimize_projections(X, Y, num_dims=num_dimensions)
X_projected = X @ projections.T

# Permutation testing for significance
num_permutations = 100
random_dist_covs = []
for _ in range(num_permutations):
    permuted_Y = np.random.permutation(Y)
    R_Y_perm = re_center_distance_matrix(compute_distance_matrix(permuted_Y))
    random_dist_covs.append(distance_covariance(R_X, R_Y_perm))

# Observed distance covariance
observed_dist_cov = distance_covariance(R_X, R_Y)
p_value = (np.sum(np.array(random_dist_covs) >=
           observed_dist_cov) + 1) / (num_permutations + 1)

print(f"Observed Distance Covariance: {observed_dist_cov}")
print(f"P-Value (Permutation Test): {p_value}")

# Combined results with enhanced visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Raw Data X and Y
axes[0, 0].imshow(X, aspect='auto', cmap='viridis')
axes[0, 0].set_title("Raw Data X", fontsize=14)
axes[0, 0].set_xlabel("Features", fontsize=12)
axes[0, 0].set_ylabel("Samples", fontsize=12)
axes[0, 0].colorbar = plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])

axes[0, 1].imshow(Y, aspect='auto', cmap='viridis')
axes[0, 1].set_title("Raw Data Y", fontsize=14)
axes[0, 1].set_xlabel("Features", fontsize=12)
axes[0, 1].set_ylabel("Samples", fontsize=12)
axes[0, 1].colorbar = plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])

# Re-centered Distance Matrices
axes[1, 0].imshow(R_X, aspect='auto', cmap='coolwarm')
axes[1, 0].set_title("Re-centered Distance Matrix for X", fontsize=14)
axes[1, 0].set_xlabel("Samples", fontsize=12)
axes[1, 0].set_ylabel("Samples", fontsize=12)
axes[1, 0].colorbar = plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0])

axes[1, 1].imshow(R_Y, aspect='auto', cmap='coolwarm')
axes[1, 1].set_title("Re-centered Distance Matrix for Y", fontsize=14)
axes[1, 1].set_xlabel("Samples", fontsize=12)
axes[1, 1].set_ylabel("Samples", fontsize=12)
axes[1, 1].colorbar = plt.colorbar(axes[1, 1].images[0], ax=axes[1, 1])

plt.tight_layout()
plt.show()

# Projections and Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_projected[:, 0], np.mean(Y, axis=1), alpha=0.7,
            label="First Projection of X vs. Y Mean", color="blue")
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.title("Optimized Projection of X vs. Mean of Y", fontsize=16)
plt.xlabel("First Projection of X", fontsize=14)
plt.ylabel("Mean of Y", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)

# Display observed distance covariance
plt.text(min(X_projected[:, 0]) * 0.9, max(np.mean(Y, axis=1)) * 0.9,
         f"Obs. Dist. Cov.: {observed_dist_cov:.2f}\nP-Value: {p_value:.4f}",
         fontsize=12, color="darkred", bbox=dict(facecolor="white", alpha=0.8))
plt.tight_layout()
plt.show()

# Histogram of Permutation Test
plt.figure(figsize=(8, 5))
plt.hist(random_dist_covs, bins=20, color='lightgray',
         edgecolor='black', alpha=0.7, label="Permuted Dist. Cov.")
plt.axvline(observed_dist_cov, color='red',
            linestyle='--', label="Observed Dist. Cov.")
plt.title("Permutation Test Results", fontsize=16)
plt.xlabel("Distance Covariance", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()
