# Install the energy package if needed
dir.create(path = "~/R/libraries", showWarnings = FALSE, recursive = TRUE)
install.packages("languageserver", lib = "~/R/libraries")
install.packages("energy",  lib = "~/R/libraries", , dependencies = TRUE)

.libPaths("~/R/libraries")
library(energy)

run_tests <- function() {
  # Standard case with two distinct groups
  group1 <- matrix(rnorm(20 * 2), nrow=20, ncol=2)
  group2 <- matrix(rnorm(20 * 2, mean=5), nrow=20, ncol=2)
  data <- rbind(group1, group2)
  groups <- factor(c(rep(1, 20), rep(2, 20)))
  result <- disco(data, groups, R=500)
  print(result)

  # Identical groups (should give high p-value)
  group1 <- matrix(rnorm(20 * 2), nrow=20, ncol=2)
  group2 <- group1
  data <- rbind(group1, group2)
  groups <- factor(c(rep(1, 20), rep(2, 20)))
  result <- disco(data, groups, R=500)
  print(result)

  # Single-point groups (expected zero within-sample dispersion)
  group1 <- matrix(c(0, 0), nrow=1)
  group2 <- matrix(c(1, 1), nrow=1)
  data <- rbind(group1, group2)
  groups <- factor(c(1, 2))
  result <- disco(data, groups, R=500)
  print(result)

  # Varied group sizes
  group1 <- matrix(rnorm(10 * 2), nrow=10, ncol=2)
  group2 <- matrix(rnorm(30 * 2, mean=5), nrow=30, ncol=2)
  data <- rbind(group1, group2)
  groups <- factor(c(rep(1, 10), rep(2, 30)))
  result <- disco(data, groups, R=500)
  print(result)

  # Large sample size
  group1 <- matrix(rnorm(100 * 2), nrow=100, ncol=2)
  group2 <- matrix(rnorm(100 * 2, mean=5), nrow=100, ncol=2)
  data <- rbind(group1, group2)
  groups <- factor(c(rep(1, 100), rep(2, 100)))
  result <- disco(data, groups, R=100)
  print(result)

  # Alpha variants (disco function only supports alpha = 1, so skip this part)

  # Three distinct groups
  group1 <- matrix(rnorm(15 * 2), nrow=15, ncol=2)
  group2 <- matrix(rnorm(15 * 2, mean=5), nrow=15, ncol=2)
  group3 <- matrix(rnorm(15 * 2, mean=10), nrow=15, ncol=2)
  data <- rbind(group1, group2, group3)
  groups <- factor(c(rep(1, 15), rep(2, 15), rep(3, 15)))
  result <- disco(data, groups, R=500)
  print(result)

  # High-dimensional data
  group1 <- matrix(rnorm(20 * 50), nrow=20, ncol=50)
  group2 <- matrix(rnorm(20 * 50, mean=5), nrow=20, ncol=50)
  data <- rbind(group1, group2)
  groups <- factor(c(rep(1, 20), rep(2, 20)))
  result <- disco(data, groups, R=100)
  print(result)

  # Zero dispersion (identical points in each group)
  group1 <- matrix(rep(1, 20 * 2), nrow=20, ncol=2)
  group2 <- matrix(rep(1, 20 * 2), nrow=20, ncol=2)
  data <- rbind(group1, group2)
  groups <- factor(c(rep(1, 20), rep(2, 20)))
  result <- disco(data, groups, R=100)
  print(result)
}

# Run the tests
run_tests()
