
CV_SVM_weight <- function(data, K, cost){
  ### Input:
  ### - data: train data
  ### - K: a number stands for K-fold CV
  ### - cost: parameters 
  ### Output: mean error and standard deviatation of the error
  set.seed(0)
  #labels = dat_train$label
  n <- dim(data)[1]
  n_fold <- round(n/K, 0)
  s <- sample(rep(1:K, c(rep(n_fold, K-1), n-(K-1) * n_fold)))  
  cv_error <- rep(NA, K)

  for (i in 1:K){
    temp <- ovun.sample(label ~ ., data = data, p = 0.3, method = "over")$data
    train_data <- temp[s != i,]
    test_data <- data[s == i,]
    par <- list(cost = cost)
    
    # To train an SVM model by using processed features from training data
    fit <- svm(label ~., data = train_data, kernel = "linear", cost = par) 
    # Generative predictions 
    pred <- predict(fit, test_data)  
    error <- mean(pred != test_data$label) 
    print(error)
    cv_error[i] <- error
    mean_cv_error <- mean(cv_error)
    sd_cv_error <- sd(cv_error)
  }			
  return(c(mean_cv_error, sd_cv_error))
}


