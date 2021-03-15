cvrf_weight.function <- function(dat_train, K, ntree){
  
  n <- dim(dat_train)[1]
  n.fold <- round(n/K, 0)
  set.seed(0)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    temp <- ovun.sample(label ~ ., data = dat_train, p = 0.5, method = "both")$data
    train_data <- temp[s != i,]
    test_data <- dat_train[s == i,]
    
    
    rf_pred <- predict(train_rf(train_data,ntree),test_data[,-which(names(test_data) == 'label')])
    
    
    error <- mean(rf_pred != test_data$label) 
    print(error)
    cv.error[i] <- error
    
  }			
  return(c(mean(cv.error),sd(cv.error)))
}