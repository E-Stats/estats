# Install the energy package if needed
dir.create(path = "~/R/libraries", showWarnings = FALSE, recursive = TRUE)
install.packages("languageserver", lib = "~/R/libraries")
install.packages("energy",  lib = "~/R/libraries", dependencies = TRUE)

.libPaths("~/R/libraries")
library(energy)

test_mvnorm_statistic <- function() {
  # Set seed for reproducibility
  set.seed(42)
  
  # Test Case 1: Multivariate normal data
  data_normal <- matrix(rnorm(100 * 5, mean=0, sd=1), ncol=5)  # 100 samples, 5 dimensions
  statistic_normal <- mvnorm.e(data_normal)
  cat("Statistic for normal data (100x5): ", statistic_normal, "\n")
  
  # Test Case 2: Non-normal data (uniform distribution)
  data_non_normal <- matrix(runif(100 * 5, min=-5, max=5), ncol=5)
  statistic_non_normal <- mvnorm.e(data_non_normal)
  cat("Statistic for uniform data (100x5): ", statistic_non_normal, "\n")
  
  # Test Case 3: Larger multivariate normal data
  data_large_normal <- matrix(rnorm(500 * 10, mean=0, sd=1), ncol=10)  # 500 samples, 10 dimensions
  statistic_large_normal <- mvnorm.e(data_large_normal)
  cat("Statistic for normal data (500x10): ", statistic_large_normal, "\n")
  
  # Test Case 4: Larger non-normal data (uniform distribution)
  data_large_non_normal <- matrix(runif(500 * 10, min=-10, max=10), ncol=10)
  statistic_large_non_normal <- mvnorm.e(data_large_non_normal)
  cat("Statistic for uniform data (500x10): ", statistic_large_non_normal, "\n")
}

# Run the test
test_mvnorm_statistic()
