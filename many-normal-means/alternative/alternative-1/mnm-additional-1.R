eva_JS <- function(ys,mus){
  n <- length(ys)
  mu_js <- mean(ys) + (1 - (n-3)/sum((ys - mean(ys))**2)) * (ys - mean(ys))
  
  return(sum((mu_js - mus)**2))
}

risk_js <- rep(NA,N)
risk_mjs <- rep(NA,N)

this.example <- c(1,2,3)
n <- 10
N <- 1000

for (example in this.example){
  
  for (n in c(10,20,50)){
    
    set.seed(123)
    
    cat("Example: ",example, " n = ",n, "\n")
    
    for (nn in 1:N){
      

      if(TRUE){
        

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
        
        
        nus <- c(seq(min(ys), max(ys), by = 0.1),0)
        ws <- rep(1/length(nus),length(nus))
        rhos <- rep(0, length(nus))
        js_mat <- matrix(NA, nrow = length(nus), ncol = n)
        for (i in 1:length(nus)){
          rhos[i] <- ws[i]*(sqrt(sum((ys - nus[i])**2))**(-(n-2)))
          js_mat[i,] <- ys - (n - 2)*(ys - nus[i])/(sum((ys - nus[i])**2))
        }

        rhos <- rhos/sum(rhos)
        mult_js <- rep(0, n)
        
        for (i in 1:length(nus)){
          mult_js <- mult_js + js_mat[i,]*rhos[i]  
        }
        
        risk_mjs[nn] <- sum((mult_js - mus)**2)
        risk_js[nn] <- eva_JS(ys,mus)
        
        mu_js <- mean(ys) + (1 - (n-3)/sum((ys - mean(ys))**2)) * (ys - mean(ys))
        
        
        if (nn == N){
          print(mean(risk_mjs)/n)
          print(mean(risk_js)/n)
        }
        
        
        
        
      }
      
    }
  }
}
