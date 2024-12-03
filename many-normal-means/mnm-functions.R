library("deconvolveR") ## g-modeling
require(quadprog)
library(dirichletprocess)

mysoftmax <- function(xs){
  this.sum <- sum(exp(xs))
  return(exp(xs)/this.sum)
}



mynll <- function(etas,alphas, ys){
  n <- length(ys)
  m <- length(alphas)
  ll <- 0
  
  for (i in 1:n){
    this.sum <- 0
    for(k in 1:m){
      this.sum <- this.sum + (alphas[k] * exp(- (1/2)*((ys[i] - etas[k])^2)))/sqrt(2*pi)
    }
    ll <- ll + log(this.sum)
  }
  return(-ll/n)
}



mypen <- function(lambdas,etas){
  m <- length(etas)
  pen <- 0
  this.diff <- c(etas[2:m],0) - etas
  return(sum(abs(lambdas * this.diff[1:(m-1)])))
}


d_eta_alpha <- function(etas,alphas,ys){
  n <- length(ys)
  m <- length(alphas)
  this.d <- 0
  this.sum <- 0
  d_alphas <- rep(0,m)
  d_etas <- rep(0,m)
  
  for(i in 1:n){
    this.sum <- 0
    for (k in 1:m){
      this.sum <- this.sum + alphas[k] * exp(- (1/2)*((ys[i] - etas[k])^2))
    }
    
    for (k in 1:m){
      d_etas[k] <- d_etas[k] + (alphas[k] * exp(- (1/2)*((ys[i] - etas[k])^2)) * (ys[i] - etas[k]))/this.sum
      
      d_alphas[k] <- d_alphas[k] + exp(- (1/2)*((ys[i] - etas[k])^2))/this.sum
    }
  }
  return(-c(d_etas,d_alphas)/n)
}



bisec_nu <- function(x,a0,b0){
  for(K in 1:100){
    if (d_nu((a0+b0)/2,x) <= 0){
      a1 <- a0
      b1 <- (a0+b0)/2
    }else{
      b1 <- b0
      a1 <- (a0+b0)/2
    }
    
    if(abs(a1 - a0) + abs(b1-b0) < 1e-4){
      break
    }
    
    a0 <- a1
    b0 <- b1
    
  }
  return(a1)
}


my_func <- function(x,v){
  this.a <- v - x
  this.a[this.a <= 0] <- 0
  return(sum(this.a) - 1)
}

projection_simplex_bisection <- function(v, tau=0.0001, max_iter=1000){
  lower <- min(v) - 1 / length(v)
  upper <- max(v)
  for (it in 1:max_iter){
    midpoint <- (upper + lower) / 2.0
    value <- my_func(midpoint,v)
    if (abs(value) <= tau){
      break
    }
    
    if (it == max_iter){
      cat("Warining: Potential Convergence Issue\n")
    }    
    
    if (value <= 0){
      upper <- midpoint
    }else{
      lower <- midpoint
    }
  }
  
  this.b <- v - midpoint
  this.b[this.b <= 0] <- 0
  return(this.b)
}

isSorted <- function(x){
  return(sum(diff(x) <= 0) == 0)
}

sort_eta <- function(etas){
  m <- length(etas)
  for (i in 1:(m-1)){
    if (etas[i] > etas[i+1]){
      etas[i] = etas[i+1]
    }
  }
  return(etas)
}


param_update <- function(lambdas,ys, max_iteration = 500, tol = 1e-6,
                         step.size = 0.05, start_etas = 0, start_alphas = 0){
  
  
  n <- length(ys)
  m <- length(lambdas)+1
  
  A <- matrix(0,nrow = m-1, ncol = m)
  for (i in 1:(m-1)){
    A[i,i] <- -1
    A[i,i+1] <- 1
  }
  
  
  d.pen <- -diff(c(0,lambdas,0))
  if (length(start_etas) == 1){
    eta0s <- sort(rnorm(m))
    alpha0s <- rep(1/m,m)
  }else{
    eta0s <- start_etas
    alpha0s <- start_alphas
  }
  
  hist <- c()
  for(ii in 1:max_iteration){
    
    
    
    this.d.pen <- d.pen
    
    
    hist <- c(hist, mynll(eta0s, alpha0s, ys) + mypen(lambdas,eta0s))
    
    this.g <- d_eta_alpha(eta0s,alpha0s,ys)
    
    new.etas <- eta0s - 10*step.size* (this.g[1:m] + this.d.pen)
    
    this.result <- solve.QP(Dmat = diag(1,nrow = m), dvec = new.etas, Amat = t(A), bvec = rep(0,m-1),meq = 0)
    
    new.etas <- this.result$solution
    
    
    new.alphas <- alpha0s - step.size*this.g[(m+1):(2*m)]
    
    new.alphas <- projection_simplex_bisection(new.alphas)
    
    
    if(ii >= 10 && abs(hist[ii] - hist[ii-1]) <= tol){
      break
    }
    
    eta0s <- new.etas
    alpha0s <- new.alphas
    
  }
  
  return(list(etas_hat = new.etas, alphas_hat = new.alphas, ll_hist = hist))
}


mnm_AM <- function(ys,m,BB = 5, K = 5, isBootstrap = T, J = 2000, max_iteration = 500, tol = 1e-6,
                   step.size = 0.05, num.steps = 1, step.size.lambda = 0.1){ 
  ## ys: data
  ## m: the number of discrete points
  ## BB: the number of imputations
  ## K: the number of fold --- resulting in BB*K imputation models
  ## J: Maximum iteration of AM (usually converges fast)
  ## max_iteration: Maximum number of iterations for estimating the initial value of parameters.
  ## tol: the convergence criteria
  ## step.size: learning rate of the model parameters
  ## step.size.lambda: learning rate of the duality parameters
  
  
  
  B <- matrix(0,nrow = m, ncol = m-1)
  for (i in 1:(m-1)){
    B[i,i] <- 1
    B[i+1,i] <- -1
  }
  BTB <- t(B) %*% B
  
  
  m.ratios <- c(0.5,0.8,1,1.2) ## Candidate set of the resampling ratio for the m-out-of-n bootstrap
  m.p.values <- rep(NA, length(m.ratios))
  
  if(T){
    
    ys_imp <- c()
    
    for (b in 1:BB){
      isRepeat <- T
      
      if (b == 1){
        this.m.ratio.index <- 0
      }else{
        this.m.ratio.index <- this.select.m.index
        this.m.ratio <- m.ratios[this.select.m.index]
      }
      
      
      indices <- sample(1:n)
      fold_size <- n %/% K
      additional_elements <- n %% K
      
      # Initialize folds
      folds <- vector("list", K)
      
      # Assign indices to each fold
      start_index <- 1
      for (k in 1:K) {
        end_index <- start_index + fold_size - 1
        
        # Distribute the remaining elements in the first few folds
        if (k <= additional_elements) {
          end_index <- end_index + 1
        }
        
        folds[[k]] <- indices[start_index:end_index]
        start_index <- end_index + 1
      }
      
      
      
      while(isRepeat){
        
        F.vector <- c()
        
        if (b == 1){
          this.m.ratio.index <- this.m.ratio.index + 1
          this.m.ratio <- m.ratios[this.m.ratio.index]
          
        }
        
        if (this.m.ratio.index > length(m.ratios)){
          break
        }
        
        cat("Current m.ratio: ", this.m.ratio, "\n")
        
        
        for(k in 1:K){
          
          validation_indices <- folds[[k]]
          
          # Training set indices are all other indices
          training_indices <- unlist(folds[-k])
          
          
          repeat_iteration <- TRUE
          
          while (repeat_iteration) {
            
            cat("Begin Imputation: ", b,"/", BB, " Fold: ",k,"/",K,"\n")
            
            
            if (isBootstrap){ ## Default: the m-out-of-n bootstrap
              
              current_indices <- sample(seq(length(training_indices)), ceiling(this.m.ratio*length(training_indices)), replace = TRUE)
              future_indices <- seq(length(training_indices))
              
              m.impute <- max(length(unique(current_indices)),2)
              
              B.impute <- matrix(0,nrow = m.impute, ncol = m.impute-1)
              
              for (i in 1:(m.impute-1)){
                
                B.impute[i,i] <- 1
                B.impute[i+1,i] <- -1
                
              }
              
              B.imputeTB.impute <- t(B.impute) %*% B.impute
              
              lambdas0 <- rep(0,m.impute-1)
              
              result <- param_update(lambdas0,ys[current_indices], max_iteration,tol,
                                     step.size)
              
              etas_hat0 <- result$etas_hat
              alphas_hat0 <- result$alphas_hat
              
              
            }else{
              
              midpoint <- floor(length(training_indices)/2)
              current_indices <- training_indices[1:midpoint]
              future_indices <- training_indices[(midpoint+1):length(training_indices)]
              
            }
            
            
            
            
            for(j in 1:J){
              
              ## Update Lambdas
              this.diff <- d_eta_alpha(etas_hat0,alphas_hat0,ys[future_indices]) - d_eta_alpha(etas_hat0,alphas_hat0,ys[current_indices])
              
              new.lambdas <- lambdas0 - step.size.lambda*(2 * (B.imputeTB.impute %*% lambdas0)[,1] + 2* (t(B.impute) %*% (-this.diff[1:m.impute]))[,1])
              new.lambdas[new.lambdas <= 0] <- 0
              
              ## Update Params
              result <- param_update(new.lambdas,ys[current_indices],max_iteration = num.steps,tol,step.size,etas_hat0, alphas_hat0)
              new.etas_hat <- result$etas_hat
              new.alphas_hat <- result$alphas_hat
              
              etas_hat0 <- new.etas_hat
              alphas_hat0 <- new.alphas_hat
              
              this.abs <- sum(abs(lambdas0 - new.lambdas))
              
              
              if (this.abs/ (sum(new.lambdas) + 1e-8) <= 1e-3){
                lambdas0 <- new.lambdas
                break
              }
              
              lambdas0 <- new.lambdas
            }
            
            if (j == J){
              
              cat("Reaching Maximum Iteration --- Start Over \n")
              
              repeat_iteration <- T
              
              
            }else{
              
              if (b == 1){
                this.F.vector <- rep(0, length(validation_indices))
                for (j in 1:length(alphas_hat0)){
                  this.F.vector <- this.F.vector + pnorm(ys[validation_indices],mean = etas_hat0[j], sd =1)*alphas_hat0[j]
                }
                F.vector <- c(F.vector, this.F.vector)
              }
              repeat_iteration <- F
              
              
              
            }
            
          }
          
          
          nn <- length(validation_indices)
          b_mus <- rep(NA,nn)
          for (i in 1:nn){
            b_mus[i] <- sample(etas_hat0, 1, pro = alphas_hat0)
          }
          this.ys_imp <- rnorm(nn,0,1) + b_mus
          ys_imp <- c(ys_imp, this.ys_imp)
          
        }
        
        if (b == 1){
          
          ks_test_result <- ks.test(F.vector, "punif", min = 0, max = 1)
          
          m.p.values[this.m.ratio.index] <- ks_test_result$p.value
        }
        
        
        
        if (b != 1){
          isRepeat = F
        }
        
        
        
      }
      
      if (b == 1){
        print(m.p.values)
        
        this.select.m.index <- which.max(m.p.values) ## Select the best m.ratio
        ys_imp <- ys_imp[(n*(this.select.m.index-1) + 1) : (n*(this.select.m.index))]
        
        cat("Selected m.ratio: ", m.ratios[this.select.m.index], "\n")
      }
      
      
      
    }
    
  }
  
  
  
  lambdas0 <- rep(0,m-1)
  
  result <- param_update(lambdas0,ys,max_iteration, tol,
                         step.size)
  
  etas_hat0 <- result$etas_hat
  alphas_hat0 <- result$alphas_hat
  
  for(j in 1:J){
    
    
    this.diff <- d_eta_alpha(etas_hat0,alphas_hat0,ys_imp) - d_eta_alpha(etas_hat0,alphas_hat0,ys)
    
    new.lambdas <- lambdas0 - step.size.lambda*(2 * (BTB %*% lambdas0)[,1] + 2* (t(B) %*% (-this.diff[1:m]))[,1])
    new.lambdas[new.lambdas <= 0] <- 0
    
    ## Update Params
    result <- param_update(new.lambdas,ys,max_iteration = num.steps,tol,step.size,etas_hat0, alphas_hat0)
    new.etas_hat <- result$etas_hat
    new.alphas_hat <- result$alphas_hat
    
    etas_hat0 <- new.etas_hat
    alphas_hat0 <- new.alphas_hat
    
    this.abs <- sum(abs(lambdas0 - new.lambdas))
    
    if (this.abs/ (sum(new.lambdas) + 1e-8) <= 1e-3){
      lambdas0 <- new.lambdas
      break
    }
    
    lambdas0 <- new.lambdas
  }
  
  
  return(result)
}


## JS estimator
eva_JS <- function(ys,mus){
  n <- length(ys)
  mu_js <- mean(ys) + (1 - (n-3)/sum((ys - mean(ys))**2)) * (ys - mean(ys))
  
  return(sum((mu_js - mus)**2))
}

## AM estimator
eva_AM <- function(result,ys,mus){
  n <- length(ys)
  etas <- result$etas_hat
  alphas <- result$alphas_hat
  risk <- rep(NA,n)
  
  for(i in 1:n){
    
    alpha_is <- rep(NA,m)
    
    for (k in 1:m){
      alpha_is[k] <- alphas[k] * exp(- (1/2)*((ys[i] - etas[k])^2)) 
    }
    
    risk[i] <- (sum((alpha_is / sum(alpha_is)) * etas) - mus[i])**2
    
  }
  
  return(sum(risk))
}


eva_gmodeling <- function(ys,mus, by = 0.1){
  mus_hat <- rep(NA,length(ys))
  tau <- c(seq(from = min(ys), to = max(ys), by = by),0)
  result <- deconv(tau = tau, X = ys, family = "Normal")
  g <- result$stats[, "g"]
  
  for(i in 1:n){
    this.prob <- exp(-(((ys[i] - tau)**2)/2)) * g
    this.prob <- this.prob/sum(this.prob)
    mus_hat[i] <- sum(this.prob*tau)
  }
  return(sum((mus_hat - mus)**2))
}




eva_dpmm <- function(ys,mus, seed = 123){
  set.seed(seed)
  mus_hat <- rep(NA,length(ys))
  
  
  dpobj <- DirichletProcessGaussianFixedVariance(ys, 1)
  dpobj <- Fit(dpobj, 1000)
  
  newData <- ys
  
  if (!is.matrix(newData))
    newData <- matrix(newData, ncol = 1)
  
  mus_hat <- rep(NA,length(newData))
  
  alpha <- dpobj$alpha
  cluster.means <- dpobj$clusterParameters[[1]][,,]
  
  
  clusterParams <- dpobj$clusterParameters
  numLabels <- dpobj$numberClusters
  mdobj <- dpobj$mixingDistribution
  
  pointsPerCluster <- dpobj$pointsPerCluster
  
  Predictive_newData <- Predictive(mdobj, newData)

  componentIndexes <- numeric(nrow(newData))
  
  for (i in seq_len(nrow(newData))) {
    
    dataVal <- newData[i, , drop = FALSE]

    weights <- pointsPerCluster * Likelihood(mdobj, dataVal, clusterParams)
    this.prob <- weights/sum(weights)
    
    mus_hat[i] <- sum(cluster.means*this.prob)
    
  }
  
  
  return(list(sum((mus_hat - mus)**2), dpobj))
}


eva_MJS <- function(ys, mus, dpobj, M = 1000, seed = 123){
  set.seed(seed)
  newData <- ys
  
  if (!is.matrix(newData))
    newData <- matrix(newData, ncol = 1)
  
  
  alpha <- dpobj$alpha
  cluster.means <- dpobj$clusterParameters[[1]][,,]
  C <- length(cluster.means)
  
  clusterParams <- dpobj$clusterParameters
  numLabels <- dpobj$numberClusters
  mdobj <- dpobj$mixingDistribution
  
  pointsPerCluster <- dpobj$pointsPerCluster
  
  Predictive_newData <- Predictive(mdobj, newData)

  
  prob.matrix <- matrix(0, nrow = n, ncol = numLabels)
  for (i in seq_len(nrow(newData))) {
    
    dataVal <- newData[i, , drop = FALSE]

    weights <- pointsPerCluster * Likelihood(mdobj, dataVal, clusterParams)
    this.prob <- weights/sum(weights)
    prob.matrix[i,] <- this.prob

    
  }
  
  log.prob.matrix <- log(prob.matrix)
  
  nus <- matrix(0, nrow = n, ncol = M)
  ws <- rep(0, M)
  
  for (m in 1:M){
    for (i in 1:n){
      
      this.index <- sample(1:numLabels, 1, prob = prob.matrix[i,])
      
      nus[i,m] <- cluster.means[this.index]
      
      ws[m] <- ws[m] + log.prob.matrix[i,this.index]
      
    }
  }
  
  ws <- exp(ws - max(ws))  # Subtract max for stability
  ws <- ws / sum(ws)
  
  rhos <- rep(0, ncol(nus))
  js_mat <- matrix(NA, nrow = ncol(nus), ncol = n)
  for (i in 1:ncol(nus)){
    rhos[i] <- ws[i]*(sqrt(sum((ys - nus[,i])**2))**(-(n-2)))
    js_mat[i,] <- ys - (n - 2)*(ys - nus[,i])/(sum((ys - nus[,i])**2))
  }
  
  rhos <- rhos/sum(rhos)
  mult_js <- rep(0, n)
  
  for (i in 1:ncol(nus)){
    mult_js <- mult_js + js_mat[i,]*rhos[i]  
  }

  return(sum((mult_js - mus)**2))
}

