###########################################################
### Make predictions with test features ###
###########################################################

### Author: Hao Hu
### Project 3
test<-function(model,features,pca,np){
  pca_test <- predict(pca,features)
  pca_test2 <- as.data.frame(pca_test)
  pca_test3 <- pca_test2[1:np]
  predict1 <- predict(model,pca_test3)
  return(predict1)
}