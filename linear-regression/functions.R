library(glmnet)
library(MASS)
library(lars)


my.sample <- function(x, size, replace=TRUE, prob=NULL, q=0){
  y <- runif(size) < q
  z <- sample(x, size=size-sum(y), replace = TRUE)

  return(c(x[y], z))
}

SoftS <- function(z, gamma = 1){
  if (z < gamma && z > -gamma){
    theta <- 0 
  }else if (z > gamma){
    theta <- z - gamma
  }else{
    theta <- z + gamma
  }
  return(theta)
}


MC_xTy <- function(X,y,this.xmean = 0,this.xsd = 1,this.ymean = 0, q = 0, m.ratio = 1, beta_mle = NULL,
                   isStandardize = TRUE, isBootstrap = TRUE, isL2 = FALSE, isSingle_lambda = F,
                   t1 = 1e-5, t2 = 1e+0, alpha_a = 1, alpha_sigma = 1, tol1 = 1e-5, tol2 = 1e-3, isVerbose = F){
  n <- dim(X)[1]
  p <- dim(X)[2]


  if(isBootstrap){

    this.index1 <- my.sample(1:n, size= max(3,ceiling(m.ratio*n)), replace=TRUE, q=q)
    left.out <- seq(n)
    X_tilde <- X[c(this.index1),]
    y_tilde <- y[c(this.index1)]
    this.index2 <- setdiff(seq(n), this.index1)
    

  }else{
    
    this.index1 <- sample(seq(n),ceiling(m.ratio*n), replace = F)
    this.index2 <- setdiff(seq(n), this.index1)
    X_tilde <- X[c(this.index1),]
    y_tilde <- y[c(this.index1)]
    X <- X[c(this.index2),]
    y <- y[c(this.index2)]

  }
  
  if(isStandardize){

    this.ymean <- mean(y_tilde)
    this.xmean <- apply(X_tilde,2,mean)
    this.xsd <-  sqrt(((nrow(X_tilde)-1)/nrow(X_tilde)) *apply(X_tilde,2,var));
    
    for (i in 1:nrow(X_tilde)){
      X_tilde[i,] <- (X_tilde[i,]- this.xmean) / this.xsd
    }
    
    y_tilde <- y_tilde - this.ymean
    
    for (i in 1:nrow(X)){
      X[i,] <- (X[i,]- this.xmean) / this.xsd
    }
    
    y <- y - this.ymean
  }

  s1 <- (t(X) %*% X/nrow(X) - t(X_tilde) %*% X_tilde/nrow(X_tilde))
  s2 <- (t(X_tilde) %*% y_tilde/nrow(X_tilde) - t(X) %*% y/nrow(X))[,1]
  
    
  

  result <- stochastic_update(X_tilde, y_tilde,  s1, s2, isL2, X, y,
                                  isSingle_lambda, t1, t2, alpha_a,alpha_sigma, tol1, tol2,isVerbose=isVerbose)


  
  beta_tilde <- result$beta_hat
  


  if(isStandardize){

    for (i in 1:nrow(X_tilde)){
      X_tilde[i,] <- (X_tilde[i,]*this.xsd) + this.xmean 
    }
    
    y_tilde <- y_tilde + this.ymean
    
    beta_tilde <- beta_tilde / this.xsd
  }
  
  e_tilde <- y_tilde - X_tilde %*% beta_tilde
  



  return(list(X_tilde = X_tilde, y_tilde = y_tilde, e_tilde = e_tilde, beta_tilde = beta_tilde,this.index1 = this.index1, sigma_sq_tilde = result$sigma_sq_hat,
              lambda_hat_zero = result$lambda_hat_zero))
}




stochastic_update <- function(X,y,s1,s2,isL2 =F, X_valid = X, y_valid = y, isSingle_lambda = F,
                              t1 = 1e-5, t2 = 1e+0, alpha_a = 1, alpha_sigma = 1, 
                              tol1 = 1e-5, tol2 = 1e-3, q = 1, min_iteration = 100, 
                              max_iteration = 50000, isVerbose = F){
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  beta_hat_0 <- rep(0,p)
  MSE_hist <- c()

  
  if (isSingle_lambda){
    
    lambda_hat <- 1e-1
    lambda_hat_zero <- (n - 1)/2
    mu_a <- 0
    nu_a <- 0
    mu_sigma <- 0
    nu_sigma <- 0
    

  }else{
    
    lambda_hat <- rnorm(p)
    lambda_hat[lambda_hat < 0] <- -lambda_hat[lambda_hat < 0]

  
    mu_a <- rep(0,p)
    nu_a <- rep(0,p)

    
  }
  
  beta1_a = 0.9
  beta2_a = 0.999
  eps = 1e-7
  rho_a = (alpha_a/(1+beta1_a))


  round <- 0
  mse_current_0 <- Inf
  mse_future_0 <- Inf

  while(TRUE){
    
    round <- round + 1
      
    if (!isL2){
      if (isSingle_lambda){
        this.mid <- beta_hat_0 + t1 * (t(X) %*% (y - (X %*% beta_hat_0)[,1]))[,1]
        beta_hat_1 <- beta_hat_0
        for (i in 1:p){
          beta_hat_1[i] <- SoftS(this.mid[i], t1*lambda_hat)
        }
      }else{
        
        this.mid <- beta_hat_0 + t1 * (t(X) %*% (y - (X %*% beta_hat_0)[,1]))[,1]
        beta_hat_1 <- beta_hat_0
        for (i in 1:p){
          beta_hat_1[i] <- SoftS(this.mid[i], t1*lambda_hat[i])

       }
      
      }
    }else{
        
        beta_hat_1 <- beta_hat_0 + t1*((t(X) %*% (y - (X %*% beta_hat_0)[,1]))[,1] - 2*(beta_hat_0)* lambda_hat)
        
    }

      this.value <- sum(abs(beta_hat_0 -beta_hat_1))/(sum(abs(beta_hat_1)) + 1e-8)

      


      if (isVerbose){
        mse_current_1 <- sum(((X %*% beta_hat_1)[,1] - y)**2)
        
        if(mse_current_1 < 1e-128){
          mse_current_1 <- 1e-128
        }
        
        mse_future_1 <- sum(((X_valid %*% beta_hat_1)[,1] - y_valid)**2)
        
        if(mse_future_1 < 1e-128){
          mse_future_1 <- 1e-128
        }
        
        MSE_hist <- c(MSE_hist, mse_future_1)
        sigma_sq_hat <- max(mse_future_1 / nrow(X_valid), mse_current_1/n)
        
        
        if (round %% 1000 == 0){
          
          if (isSingle_lambda){
            cat("Iteration ", round, " MSE-future: ", mse_future_1/nrow(X_valid)," MSE-current: ", mse_current_1/n," Beta_diff: ",this.value, " Lambda: ", lambda_hat, " Sigma_sq: ",sigma_sq_hat, "\n")
          }else{
            cat("Iteration ", round, " MSE-future: ", mse_future_1/nrow(X_valid)," MSE-current: ", mse_current_1/n," Beta_diff: ",this.value, " Sigma_sq: ",sigma_sq_hat, "\n")
            
          }
        }
      }
      

      
      if(round >= min_iteration){
        
        if (this.value <= tol1){
          mse_current_1 <- sum(((X %*% beta_hat_1)[,1] - y)**2)
          mse_future_1 <- sum(((X_valid %*% beta_hat_1)[,1] - y_valid)**2)
          sigma_sq_hat <- max(mse_future_1 / nrow(X_valid), mse_current_1/n)
          
          
          
          
          
          
          
          
          
          if (isVerbose){
            plot(MSE_hist)
          }
          
          break
        }
      }
      
      
      if(round >= max_iteration){
          mse_current_1 <- sum(((X %*% beta_hat_1)[,1] - y)**2)
          mse_future_1 <- sum(((X_valid %*% beta_hat_1)[,1] - y_valid)**2)
          sigma_sq_hat <- max(mse_future_1 / nrow(X_valid), mse_current_1/n)
        if (isVerbose){
          plot(MSE_hist)
        }
        break
      }
      

      beta_hat_0 <- beta_hat_1

      if (!isL2){
      
      if (isSingle_lambda){
        
   
        Z <- 2*sum(sign(beta_hat_1) ** 2)*lambda_hat - 2 * sum(n*((s1 %*% beta_hat_1)[,1] + s2) * sign(beta_hat_1))
        mu_a <- beta1_a * mu_a + (1- beta1_a) *Z 
        nu_a <- beta2_a * nu_a + (1- beta2_a) *Z*Z
        lambda_hat <- lambda_hat - rho_a * mu_a/(sqrt(nu_a/(1-beta2_a))+eps)
        
        if (lambda_hat < 0){
          lambda_hat <- 0
        }
        

      
      }else{
        

        Z <- - (( n*((s1 %*% beta_hat_1)[,1] + s2) - sign(beta_hat_1)* lambda_hat) *2 *sign(beta_hat_1))
        mu_a <- beta1_a * mu_a + (1- beta1_a) *Z 
        nu_a <- beta2_a * nu_a + (1- beta2_a) *Z*Z
        lambda_hat <- lambda_hat - rho_a * mu_a/(sqrt(nu_a/(1-beta2_a))+eps)
        lambda_hat[lambda_hat < 0] <- 0
       
      }
      
      }else{
        if (isSingle_lambda){
          
          Z <- 2*4*sum((beta_hat_1)**2)*lambda_hat - 2 * sum(n*((s1 %*% beta_hat_1)[,1] + s2) * 2*(beta_hat_1))
          mu_a <- beta1_a * mu_a + (1- beta1_a) *Z 
          nu_a <- beta2_a * nu_a + (1- beta2_a) *Z*Z
          
          lambda_hat <- lambda_hat - rho_a * mu_a/(sqrt(nu_a/(1-beta2_a))+eps)
          
          if (lambda_hat < 0){
            lambda_hat <- 0
          }
          
        }else{
          
          Z <- - ((n*((s1 %*% beta_hat_1)[,1] + s2) - 2*beta_hat_1 * lambda_hat) *4 * beta_hat_1)
          mu_a <- beta1_a * mu_a + (1- beta1_a) *Z 
          nu_a <- beta2_a * nu_a + (1- beta2_a) *Z*Z
          
          lambda_hat <- lambda_hat - rho_a * mu_a/(sqrt(nu_a/(1-beta2_a))+eps)
          lambda_hat[lambda_hat < 0] <- 0
          
        }
        
      }
      
      
      
      
      
  }
    
  
  
  return(list(beta_hat = beta_hat_1, sigma_sq_hat = sigma_sq_hat, lambda_hat = lambda_hat))
}


get_FN_FP <- function(true_set, pred_set){
  FN <- sum(!(true_set %in% pred_set))
  FP <- sum(!(pred_set %in% true_set))
  return(c(FN,FP))
}

get_recall <- function(beta_hat, beta = beta){
  return(length(intersect(which(beta_hat != 0), which(beta != 0))))
}

