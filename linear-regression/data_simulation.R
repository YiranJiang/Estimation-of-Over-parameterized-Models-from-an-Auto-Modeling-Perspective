library("MASS")
library("LaplacesDemon")

simulate_linear_regression <- function(p, n, rho, alpha, tau) {
  # Generating the covariance matrix Sigma
  Sigma <- matrix(rho, nrow = p, ncol = p)
  diag(Sigma) <- 1
  
  # Generating the design matrix X
  X <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  
  # Determining the number of non-zero elements in beta
  num_nonzero <- ceiling(n^alpha)
  
  # Randomly selecting non-zero elements for beta
  nonzero_indices <- sample(p, num_nonzero)
  beta <- rep(0, p)
  beta[nonzero_indices] <- rlaplace(num_nonzero, 0, 1)
  
  # Calculating error variance
  sigma_squared <- (1 / tau) * crossprod(beta, Sigma %*% beta)
  
  # Generating the error term epsilon
  epsilon <- rnorm(n, mean = 0, sd = sqrt(sigma_squared))
  
  # Computing the response vector y
  y <- X %*% beta + epsilon
  
  return(list(X = X, y = y, beta = beta, sigma = sqrt(sigma_squared)))
}

set.seed(123)
K <- 100
n <- 100
p <- 500
for (alpha in c(0.3, 0.6, 0.9)){
  for (tau in c(0.3,1,3))
    for (rho in c(0.5)){
      for (k in 1: K){
        rho <- 0.5
        result <- simulate_linear_regression(p, n, rho, alpha, tau)
        beta <- result$beta
        y <- result$y
        X <- result$X
        sigma <- result$sigma
        
        
        save(beta, X, y, sigma, file = paste0("simulated-data/alpha-",alpha,"_",k,"-tau_",tau,".RData"))
      }
    }
}
