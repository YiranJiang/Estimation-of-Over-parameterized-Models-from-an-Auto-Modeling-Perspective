library("natural")
source("functions.R")

args <- commandArgs(trailingOnly = TRUE)

alpha <- as.numeric(args[1])
k <- as.integer(args[2])
tau <- as.numeric(args[3])

rho <- 0.5


isSimulation <- T

n <- 100
p <- 500

## AM
K <- 5    # Number of folds
M <- 5


beta.matrix <- matrix(NA, nrow = 8, ncol = p)
mse.matrix <- matrix(NA, nrow = 8, ncol = 2) 
sd.diff.matrix <- matrix(NA, nrow = 6, ncol = 1) 


if(isSimulation){

  cat(paste0("Loading data ", "simulated-data/alpha-",alpha,"_",k,"-tau_",tau,".RData"))
  load(paste0("simulated-data/alpha-",alpha,"_",k,"-tau_",tau,".RData"))
  

}else{
  stop()
}

if(T){
  this.ymean <- mean(y)
  this.xmean <- apply(X,2,mean)
  this.xsd <-  sqrt(((n-1)/n) *apply(X,2,var));
  this.ysd <- sd(y)

  for (i in 1:n){
    X[i,] <- (X[i,]- this.xmean) / this.xsd
  }
  if (T){
    y <- (y - this.ymean)/this.ysd
  }else{
    y <- (y - this.ymean)
    this.ysd <- 1
  }
  
}

Sigma <- matrix(rho, nrow = p, ncol = p)
diag(Sigma) <- 1

set.seed(123)
method.num <- 0

M <- 5

imp_mean_matrix <- matrix(NA,nrow = n, ncol = M)
imp_var_matrix <- matrix(NA,nrow = n, ncol = M)
N <- 10
q <- 1

m.ratios <- c(0.1,0.2,0.3,0.5)

F.matrix <- matrix(NA, nrow = length(m.ratios), ncol = n)

all.orders <- c()

for (isL2 in c(F,T)){
  m.p.values <- rep(NA, length(m.ratios))

    cat("=======================\n")
    cat("AM: isL2--",isL2, "\n")
    
    
    E_xTy <- rep(0,p)
    all.orders <- c()
    all.y.imps <- c()
    
    for(m in 1:M){
      
    
      isRepeat <- T
      
      if (m == 1){
        this.m.ratio.index <- 0
      }else{
        this.m.ratio.index <- this.select.m.index
        this.m.ratio <- m.ratios[this.select.m.index]
      }
      
    
      
      indices <- sample(1:n)
      isInterrupted <- F
      redo <- 0
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
        if (m == 1){
          this.m.ratio.index <- this.m.ratio.index + 1
          this.m.ratio <- m.ratios[this.m.ratio.index]
          
        }
        
        if (this.m.ratio.index > length(m.ratios)){
          break
        }
        
        cat("Current m.ratio: ", this.m.ratio, "\n")
        
        
      for (k in 1:K){
      # Validation set indices for the k-th fold
      validation_indices <- folds[[k]]
      all.orders <- c(all.orders, rep(validation_indices,N))
      
      # Training set indices are all other indices
      training_indices <- unlist(folds[-k])
      

      this.return <- MC_xTy(X[training_indices,],y[training_indices],this.xmean,this.xsd,this.ymean,q = 0, m.ratio = this.m.ratio, NULL,
                            T,isBootstrap = T,isL2,isSingle_lambda = T,
                            t1 = 1e-5, t2 = 1e-4, alpha_a = 1, alpha_sigma = 0.5, tol1 = 1e-4,tol2 = 1e-2, 
                            isVerbose = F)
      
      # cat(ceiling(length(training_indices)*this.m.ratio)/2, this.return$lambda_hat_zero, 
      #     sqrt(this.return$sigma_sq_tilde)*this.ysd, "\n")
      
      

      this.X_tilde <- this.return$X_tilde
      this.y_tilde <- this.return$y_tilde

      imp_mean_matrix[validation_indices, m] <- X[validation_indices,] %*% this.return$beta_tilde
      imp_var_matrix[validation_indices, m] <- this.return$sigma_sq_tilde
      
      
      this.y_imp <- c()
      
      for (j in 1:N){
        this.y_imp <- c(this.y_imp,
                        X[validation_indices,] %*% this.return$beta_tilde 
                        + sqrt(this.return$sigma_sq_tilde) * rnorm(length(validation_indices)))
        
      }
      
      

      residuals <- y[validation_indices] - (X[validation_indices,] %*% this.return$beta_tilde)[,1]
      
      F.vector <- c(F.vector, pnorm(residuals,0,sqrt(this.return$sigma_sq_tilde)))

    
      repeat_iteration <- FALSE

      if(!isSimulation){
        this.y_imp[this.y_imp*this.ysd + this.ymean <= 0.5] <- (0 - this.ymean)/this.ysd
        this.y_imp[this.y_imp*this.ysd + this.ymean > 0.5] <- (1 - this.ymean)/this.ysd
      }
      all.y.imps <- c(all.y.imps, this.y_imp)
        
      
      }
        
        
        this.test <- ks.test(F.vector, "punif", min = 0, max = 1)
        cat("p-value: ", this.test$p.value, "\n")

        
        if (m == 1){
          m.p.values[this.m.ratio.index] <- this.test$p.value
          F.matrix[this.m.ratio.index, ] <- F.vector

        }
        
          
        
        if (m != 1){
          isRepeat <- F
        }
      
      
      }
      
      if (m == 1){
        this.select.m.index <- which.max(m.p.values)
        all.orders <- all.orders[(n*(this.select.m.index-1)*N + 1) : (n*(this.select.m.index)*N)]
        all.y.imps <- all.y.imps[(n*(this.select.m.index-1)*N + 1) : (n*(this.select.m.index)*N)]
        
        print(length(all.orders))
        print(length(all.y.imps))
        
        cat("Selected m.ratio: ", m.ratios[this.select.m.index], "\n")
        residual.vector <- F.matrix[this.select.m.index, ]

      }
      
      
      
      if(m == M){

        s1 <- matrix(0, nrow = p, ncol = p)
        s2 <- q*((t(X) %*% y/n)[,1] - (t(X[all.orders,]) %*% all.y.imps/(m*n*N)))
  

      
      result_1 <- stochastic_update(X, y,  s1, s2, isL2 = isL2, X_valid = X[all.orders,], y_valid = all.y.imps,
                                    isSingle_lambda = F, t1 = 1e-6, t2 = 1e-4, alpha_a = 1, tol1 = 1e-5, tol2 = 1e-8, q= q,
                                    max_iteration = 1e+5,
                                    isVerbose =  F)
      
      
     
      result_2 <- stochastic_update(X, y,  s1, s2, isL2 = isL2, X_valid = X[all.orders,], y_valid = all.y.imps,
                                    isSingle_lambda = T, t1 = 1e-6, t2 = 1e-4, alpha_a = 1, tol1 = 1e-5, tol2 = 1e-8, q= q,
                                    max_iteration = 1e+5,
                                    isVerbose =  F)
      
      
      
      for (result in list(result_1, result_2)){
        
        method.num <- method.num + 1
        
        beta_hat <- result$beta_hat * this.ysd/this.xsd
        sigma_hat <- this.ysd*sqrt(result$sigma_sq_hat)
      
      
      
        print((t(beta_hat - beta) %*% Sigma %*% (beta_hat - beta))[1,1])

        mse.matrix[method.num, ] <- c((t(beta_hat - beta) %*% Sigma %*% (beta_hat - beta))[1,1], NULL)
        sd.diff.matrix[method.num,1] <- sigma - sigma_hat
        
      
      
      beta.matrix[method.num, ] <- beta_hat

      
      }
      
      }
      
    }
      


    
  
}

set.seed(123)


## Other Methods:

## beta_1: Adaptive Lasso
## beta_2: Lasso
## beta_3: Elastic Net
## beta_4: Ridge

if(T){

fit1.cv <- cv.glmnet(X, y, alpha = 0, standardize = FALSE, intercept = FALSE)


fit <- glmnet(X, y, alpha = 0, lambda = fit1.cv$lambda.min, standardize = FALSE, intercept=FALSE)


this.mat <- coef(fit)
this.df <- data.frame( predict_names = rownames(this.mat),
                       coef_vals = matrix(this.mat))
beta_1 <- this.df$coef_vals
beta_1 <- this.df$coef_vals[2:(p+1)]
beta_4 <- beta_1

weights <- 1 / abs(beta_1) # Exclude intercept

cv_adaptive_lasso <- cv.glmnet(X, y, alpha = 1, standardize = FALSE, intercept = FALSE, penalty.factor = weights)


fit <- glmnet(X, y, alpha = 1, lambda = cv_adaptive_lasso$lambda.min, standardize = FALSE, intercept=FALSE)


this.mat <- coef(fit)
this.df <- data.frame( predict_names = rownames(this.mat),
                       coef_vals = matrix(this.mat))
beta_1 <- this.df$coef_vals
beta_1 <- this.df$coef_vals[2:(p+1)]



fit2.cv <- cv.glmnet(X, y, alpha = 1, standardize = FALSE, intercept = FALSE)


fit <- glmnet(X, y, alpha = 1, lambda = fit2.cv$lambda.min, standardize = FALSE, intercept=FALSE)

this.mat <- coef(fit)
this.df <- data.frame( predict_names = rownames(this.mat),
                       coef_vals = matrix(this.mat))
beta_2 <- this.df$coef_vals
beta_2 <- this.df$coef_vals[2:(p+1)]

# Elastic Net
this.lambda.min <- 1
this.alpha.min <- 1
max.cvm <- Inf

alphas <- seq(0,1,length.out = 101)

for (alpha in alphas){
  fit3.cv <- cv.glmnet(X, y, alpha = alpha, standardize = FALSE, intercept = FALSE)
  if (min(fit3.cv$cvm) < max.cvm){
    max.cvm <- min(fit3.cv$cvm)
    this.lambda.min <- fit3.cv$lambda.min
    this.alpha.min <- alpha
  }
}

fit <- glmnet(X, y, alpha = this.alpha.min, lambda = this.lambda.min, standardize = FALSE,intercept=FALSE)

this.mat <- coef(fit)
this.df <- data.frame( predict_names = rownames(this.mat),
                       coef_vals = matrix(this.mat))
beta_3 <- this.df$coef_vals
beta_3 <- this.df$coef_vals[2:(p+1)]

}


beta.matrix[5, ] <- beta_1*this.ysd/this.xsd
beta.matrix[6, ] <- beta_2*this.ysd/this.xsd
beta.matrix[7, ] <- beta_3*this.ysd/this.xsd
beta.matrix[8, ] <- beta_4*this.ysd/this.xsd

mse.matrix[5, ] <- c((t(beta - beta_1) %*% Sigma %*% (beta - beta_1))[1,1],NULL)
mse.matrix[6, ] <- c((t(beta - beta_2) %*% Sigma %*% (beta - beta_2))[1,1],NULL)
mse.matrix[7, ] <- c((t(beta - beta_3) %*% Sigma %*% (beta - beta_3))[1,1],NULL)
mse.matrix[8, ] <- c((t(beta - beta_4) %*% Sigma %*% (beta - beta_4))[1,1],NULL)


this.fit <- nlasso_cv(x = X, y = y)

## Compute the Standard Deviation:

sd.diff.matrix[5:6,1] <- c(sigma - this.fit$sig_obj*this.ysd, sigma - this.fit$sig_df*this.ysd)



save(beta.matrix,mse.matrix,sd.diff.matrix,residual.vector, file = paste0("./output/results-alpha-",as.numeric(args[1]),"_",as.integer(args[2]),"-tau_",tau,".RData"))


print(mse.matrix)
print(sd.diff.matrix) 

