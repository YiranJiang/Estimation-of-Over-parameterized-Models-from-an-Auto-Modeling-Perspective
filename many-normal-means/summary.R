bootstrap.sd <- function(vec, B = 10000, this.seed = 123){
  set.seed(this.seed)
  this.means <- rep(NA, B)
  for (b in 1:B){
    new.vec <- vec[sample(seq(length(vec)),length(vec),replace = T)]
    this.means[b] <- mean(new.vec)
  }
  return(sd(this.means))
}


mean.matrix <- matrix(NA, nrow = 4, ncol = 9)
sd.matrix <- matrix(NA, nrow = 4, ncol = 9)
this.col <- 0
for (this.example in c(1,2,3)){
  for (n in c(10, 20, 50)){
    
    this.col <- this.col + 1
    
    this.result <- readRDS(file = paste0("./output/n_", n, "_example_", this.example, ".RDS"))
    
    mean.matrix[,this.col] <- c(mean(this.result$risk_mle)/n,
                                mean(this.result$risk_js)/n,
                                mean(this.result$risk_gmodeling)/n,
                                mean(this.result$risk_am)/n)
    
    sd.matrix[,this.col] <- c(bootstrap.sd(this.result$risk_mle/n),
                              bootstrap.sd(this.result$risk_js/n),
                              bootstrap.sd(this.result$risk_gmodeling/n),
                              bootstrap.sd(this.result$risk_am/n))
    
  }
}


mean.matrix <- round(mean.matrix,3)
sd.matrix <- round(sd.matrix,3)
cat(mean.matrix)
cat(sd.matrix)
saveRDS(list(mean.matrix = mean.matrix, sd.matrix = sd.matrix), "./output/summary.RDS")