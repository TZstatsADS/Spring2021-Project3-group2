########################
### Cross Validation ###
########################

### Author: Yiran Lin
### Project 3

# cv.function <- function(features_train,features_test,labels_train,label_test,np){
#   nplist <- c(350,500,650,800,950,1100,1250,1400,1550,1700,1850,2000,2150,2300,2400)
#   for (i in nplist){
#     lt <-train(features,label,i)
#     prob_pred <- test(lt[1])
#   }
# }
cv.function <- function(features, labels, K,pca, np, reweight = FALSE){
  
  set.seed(2020)
  n <- dim(features)[1]
  n.fold <- round(n/K, 0)
  s <- sample(n) %% K + 1
  # cv.error <- rep(NA, K)
  # cv.AUC <- rep(NA, K)
  cv.accu <-rep(NA,K)
  
  for (i in 1:K){
    ## create features and labels for train/test
    feature_train <- features[s != i,]
    feature_test <- features[s == i,]
    label_train <- labels[s != i]
    label_test <- labels[s == i]
    
    ## sample reweighting
    # weight_train <- rep(NA, length(label_train))
    # weight_test <- rep(NA, length(label_test))
    # for (v in unique(labels)){
      # weight_train[label_train == v] = 0.5 * length(label_train) / length(label_train[label_train == v])
      # weight_test[label_test == v] = 0.5 * length(label_test) / length(label_test[label_test == v])
    # }
    # weight_train <- 0.5*table(label_train)
    # weight_test <-0.5*table(label_test)
    # print("checkpoint1",i)
    ## model training
    label_train_binary = ifelse(label_train == 2, 1, 0)
    model_train <- train(feature_train, label_train_binary,pca,np,w=NULL)
    # print("checkpoint2",i)
    ## make predictions
    prob_pred <- test(model_train,feature_test,pca,np)
    # print("checkpoint3",i)
    guess <- as.numeric(prob_pred)
    guess <- ifelse(guess == 2, 1, 0)
    # print(guess)
    label_test_binary <- ifelse(label_test == 2, 1, 0)
    accu=sum(guess == label_test_binary)/length(label_test_binary)
    # cv.error[i] <- 1 - sum(weight_test * (guess == label_test_binary)) / sum(weight_test)
    cv.accu[i]<- sum(guess == label_test_binary)/length(label_test_binary)
    print(sum(guess == label_test_binary)/length(label_test_binary))
    # tpr.fpr <- WeightedROC(guess, label_test_binary, weight_test)
    # cv.AUC[i] <- WeightedAUC(tpr.fpr)
  }
  # return(c(mean(cv.error),sd(cv.error), mean(cv.AUC), sd(cv.AUC),mean(cv.accu),sd(cv.accu)))
  return(c(mean(cv.accu),sd(cv.accu)))
}