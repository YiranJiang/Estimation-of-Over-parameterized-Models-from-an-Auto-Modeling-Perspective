source("mnm-functions.R")

folder_name <- "output" 

# Check if the folder already exists
if (!dir.exists(folder_name)){
# Create the folder
  dir.create(folder_name)
}


#Setup
if(TRUE){
  this.max_iteration <- 500
  this.tol <- 1e-6
  this.step.size <- 0.05
  this.num.steps <- 1
  this.step.size.lambda <- 0.1
  N <- 200
}


## Generate Training/Testing Data
## params: etas: 1:m, alphas 1:m

risk_am <- rep(NA,N)
risk_js <- rep(NA,N)
risk_gmodeling <- rep(NA,N)
risk_mle <- rep(NA,N)

args <- commandArgs(trailingOnly = TRUE)
this.example <- as.integer(args[1])

for (example in this.example){
  
  for (n in c(10,20,50)){
    
    set.seed(123)
    
    cat("Example: ",example, " n = ",n, "\n")
    
    for (nn in 1:N){
      
      cat("Round: ",nn, "\n")
      
      if(TRUE){
        
        m <- n ## Choose the number of discrete point to be equal to the number of datapoints --- Overparameterized
        
        if (example == 1){
          mus <- rep(0,n)
          mus <- mus + rnorm(n,0,0.1)
          ys <- rnorm(n,0,1) + mus
        }
        
        if (example == 2){
          mus <- rep(-2,n)
          mus[sample(seq(n),n/2,replace = FALSE)] <- 2
          
          mus <- mus + rnorm(n,0,0.1)
          
          ys <- rnorm(n,0,1) + mus
        }
        
        if (example == 3){
          mus <- rep(-3,n)
          mus <- mus + rnorm(n,0,1)
          mus[runif(n) < 0.9] <- 0
          
          ys <- rnorm(n,0,1) + mus
        }
        
        start_time <- Sys.time()
        
        this.result <- mnm_AM(ys,m, K = 5, BB = 5, isBootstrap = T, J = 2000, max_iteration = this.max_iteration, 
                              tol = this.tol, step.size = this.step.size, num.steps = this.num.steps, step.size.lambda = this.step.size.lambda)
        
        
        
        end_time <- Sys.time()
        time_taken <- end_time - start_time
        print(time_taken)
        
        risk_am[nn] <- eva_AM(this.result,ys,mus)
        risk_gmodeling[nn] <- eva_gmodeling(ys,mus,m)
        risk_js[nn] <- eva_JS(ys,mus)
        risk_mle[nn] <- sum((ys -mus)**2)
        
        if (nn %% 20 == 0){
          cat("Round ",nn," :\nMLE: ", round(mean(risk_mle, na.rm = T)/n, 3),"\nAM: ", round(mean(risk_am, na.rm = T)/n,3),
              "\nJS: ", round(mean(risk_js, na.rm = T)/n,3), "\ng-modeling: ",round(mean(risk_gmodeling, na.rm = T)/n,3),"\n=====\n")
        }
        
        if (nn %% 20 == 0){
          
          saveRDS(list(risk_am = risk_am, risk_js = risk_js, risk_gmodeling = risk_gmodeling,
                       risk_mle = risk_mle, ys = ys, mus = mus), paste0("./output/n_",n,"_example_",example,".RDS"))
          
          
        }
        
        
      }
      
    }
  }
}