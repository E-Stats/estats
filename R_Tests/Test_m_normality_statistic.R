# Install and load the energy package
dir.create(path = "~/R/libraries", showWarnings = FALSE, recursive = TRUE)
install.packages("languageserver", lib = "~/R/libraries")
install.packages("energy", lib = "~/R/libraries", dependencies = TRUE)

.libPaths("~/R/libraries")
library(energy)
test_mvnorm_statistic <- function() {
  set.seed(42) # Ensure reproducibility
  
  test_cases <- list(
    # Normal distributions with various shapes and sizes
    list(name = "Normal data (100x5)", data = matrix(rnorm(100 * 5), ncol=5)),
    list(name = "Normal data (500x10)", data = matrix(rnorm(500 * 10), ncol=10)),
    list(name = "Normal data (50x50)", data = matrix(rnorm(50 * 50), ncol=50)),
    list(name = "Large normal data (10000x20)", data = matrix(rnorm(10000 * 20), ncol=20)),
    
    # Non-normal distributions (uniform, binomial, etc.)
    list(name = "Uniform data (100x5)", data = matrix(runif(100 * 5, -5, 5), ncol=5)),
    list(name = "Uniform data (500x10)", data = matrix(runif(500 * 10, -10, 10), ncol=10)),
    list(name = "Large uniform data (10000x20)", data = matrix(runif(10000 * 20, -20, 20), ncol=20)),
    
    # Edge cases with very small or very high dimensional data
    list(name = "Edge Case: Single observation (1x5)", data = matrix(rnorm(1 * 5), ncol=5)),
    list(name = "Edge Case: Single dimension (100x1)", data = matrix(rnorm(100 * 1), ncol=1)),
    list(name = "Edge Case: Small dataset (2x2)", data = matrix(rnorm(2 * 2), ncol=2)),
    
    # Multimodal distributions (Gaussian mixture, bimodal)
    list(name = "Bimodal data (100x5)", data = rbind(matrix(rnorm(50 * 5, mean=-5), ncol=5), matrix(rnorm(50 * 5, mean=5), ncol=5))),
    list(name = "Gaussian Mixture (100x5)", data = rbind(matrix(rnorm(50 * 5), ncol=5), matrix(rnorm(50 * 5, mean=5), ncol=5))),
    
    # Skewed data
    list(name = "Skewed data (100x5)", data = matrix(rchisq(100 * 5, df=2), ncol=5)),
    
    # Large dataset with varied shapes
    list(name = "Large skewed data (10000x20)", data = matrix(rchisq(10000 * 20, df=2), ncol=20)),
    
    # Identical data (all values the same)
    list(name = "Identical data (100x5)", data = matrix(rep(7, 100 * 5), ncol=5)),
    
    # High-dimensional data
    list(name = "High-dimensional normal data (100x50)", data = matrix(rnorm(100 * 50), ncol=50)),
    list(name = "High-dimensional uniform data (100x50)", data = matrix(runif(100 * 50, -5, 5), ncol=50)),
    
    # Very high-dimensional data (large number of dimensions)
    list(name = "Very high-dimensional data (100x100)", data = matrix(rnorm(100 * 100), ncol=100))
  )
  
  for (case in test_cases) {
    cat("Testing case: ", case$name, "\n")
    tryCatch({
      statistic <- mvnorm.e(case$data)
      cat("Statistic for ", case$name, ": ", statistic, "\n", sep="")
    }, error = function(e) {
      cat("Failed to compute statistic for ", case$name, ": ", e$message, "\n", sep="")
    })
  }
}

# Run the updated test cases
test_mvnorm_statistic()
