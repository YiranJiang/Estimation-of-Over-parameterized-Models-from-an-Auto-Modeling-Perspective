library("MASS")

result.mse.matrix <- matrix(NA, nrow = 8, ncol = 9)
result.mse.sd.matrix <- matrix(NA, nrow = 8, ncol = 9)

result.sd.matrix <- matrix(NA, nrow = 6, ncol = 9)
result.sd.sd.matrix <- matrix(NA, nrow = 6, ncol = 9)

result.coverage.matrix <- matrix(NA,nrow = 8,ncol = 9)
result.coverage.sd.matrix <- matrix(NA,nrow = 8,ncol = 9)

bootstrap.sd <- function(vec, B = 10000, this.seed = 123){
  set.seed(this.seed)
  this.means <- rep(NA, B)
  for (b in 1:B){
    new.vec <- vec[sample(seq(length(vec)),length(vec),replace = T)]
    this.means[b] <- mean(new.vec)
  }
  return(sd(this.means))
}

# this.col <- 0
set.seed(123)
this.col<- 0
rho <- 0.5
p <- 500
n <- 1000
K <- 100
Sigma <- matrix(rho, nrow = p, ncol = p)
diag(Sigma) <- 1
X_new <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma)

for (alpha in c(0.3,0.6,0.9)){
  for(tau in c(0.3,1,3)){
    cat("alpha = ", alpha, ", tau = ", tau, "\n")
    this.col <- this.col + 1
    # num_nonzero_beta <- 500
    this.mse.matrix <- matrix(NA,nrow = 8, ncol = K)
    this.sd.matrix <- matrix(NA,nrow = 6, ncol = K)
    this.coverage.matrix <- matrix(NA,nrow = 8, ncol = K)
    # this.sd
    
    # sigma <- 1
    this.sd.diff <- rep(0,6)
    for (k in 1:K){
      
      load(paste0("output/results-alpha-",alpha,"_",k,"-tau_",tau,".RData"))
      load(paste0("simulated-data/alpha-",alpha,"_",k,"-tau_",tau,".RData"))
      y_new <- (X_new %*% beta + sigma[1,1]*rnorm(nrow(X_new)))[,1]
      
      for (j in 1:8){
        
        y_new_predict <- (X_new %*% beta.matrix[j,])[,1]
        
        if (j <= 4){
          sigma_hat <- sigma[1,1]-sd.diff.matrix[j,1]
        }else{
          sigma_hat <- sigma[1,1]-sd.diff.matrix[6,1]
        }
        
        count <- 0
        for (i in 1:nrow(X_new)){
          this.interval <- qnorm(c(0.025,0.975),mean = y_new_predict[i], sd = sigma_hat)
          if (y_new[i] <= this.interval[2] && y_new[i] >= this.interval[1]){
            count <- count+1
          }
          
        }
        
        
        this.coverage.matrix[j,k] <- count/nrow(X_new)
        
      }
      
      
      
      
      this.mse.matrix[,k] <- mse.matrix[,1]
      this.sd.matrix[,k] <- (sigma[1,1]-sd.diff.matrix[,1])/sigma[1,1]

      
    }
    
    
    result.coverage.matrix[,this.col] <- c(round(apply(this.coverage.matrix,1,mean),3))
    result.coverage.sd.matrix[,this.col] <- round(apply(this.coverage.matrix,1,bootstrap.sd),3)
    result.mse.matrix[,this.col] <- c(round(apply(this.mse.matrix,1,mean),3))
    result.sd.matrix[,this.col] <- c(round(apply(this.sd.matrix,1,mean),3))
    result.mse.sd.matrix[,this.col] <- round(apply(this.mse.matrix,1,bootstrap.sd),3)
    result.sd.sd.matrix[,this.col] <- round(apply(this.sd.matrix,1,bootstrap.sd),3)
  } 
  
  
}

round(result.mse.matrix,2)
round(result.mse.sd.matrix,2)
result.sd.matrix
result.coverage.matrix
result.coverage.sd.matrix