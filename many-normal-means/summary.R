bootstrap.sd <- function(vec, B = 10000, this.seed = 123){
  set.seed(this.seed)
  this.means <- rep(NA, B)
  for (b in 1:B){
    new.vec <- vec[sample(seq(length(vec)),length(vec),replace = T)]
    this.means[b] <- mean(new.vec)
  }
  return(sd(this.means))
}


mean.matrix <- matrix(NA, nrow = 6, ncol = 9)
sd.matrix <- matrix(NA, nrow = 6, ncol = 9)


this.col <- 0
for (this.example in c(1,2,3)){
  for (n in c(10, 20, 50)){
    # for (n in c(10)){
    v1 <- c()
    v2 <- c()
    v3 <- c()
    v4 <- c()
    v5 <- c()
    v6 <- c()
    
    this.col <- this.col + 1
    

    for (this.id in 1:50){
      
      print(this.id)
      
      this.result <- readRDS(file = paste0("./output-1112/n_", n, "_example_", this.example, "_",this.id,".RDS"))
      this.result$risk_am
      v1 <- c(v1,na.omit(this.result$risk_mle))    
      v2 <- c(v2,na.omit(this.result$risk_js))    
      v3 <- c(v3,na.omit(this.result$risk_mjs))    
      v4 <- c(v4,na.omit(this.result$risk_gmodeling))    
      v5 <- c(v5,na.omit(this.result$risk_dpmm))    
      v6 <- c(v6,na.omit(this.result$risk_am))    
      
    }
    
    
    mean.matrix[,this.col] <- c(mean(v1)/n,
                                mean(v2)/n,
                                mean(v3)/n,
                                mean(v4)/n,
                                mean(v5)/n,
                                mean(v6)/n)
    

    
  }
}


mean.matrix <- round(mean.matrix,3)
sd.matrix <- round(sd.matrix,3)


colnames(mean.matrix) <- rep(c("n = 10", "n = 20", "n = 50"),3)
rownames(mean.matrix) <- c("MLE", "JS", "MJS","g-modeling", "DPMM", "AM")
colnames(sd.matrix) <- rep(c("n = 10", "n = 20", "n = 50"),3)
rownames(sd.matrix) <- c("MLE", "JS", "MJS","g-modeling", "DPMM", "AM")

print(mean.matrix[c(1,2,3,5,4,6),])
print(sd.matrix[c(1,2,3,5,4,6),])


saveRDS(list(mean.matrix = mean.matrix, sd.matrix = sd.matrix), "./output/summary.RDS")


