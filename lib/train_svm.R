###########################################################
### Train a classification model with training features ###
###########################################################

### Author: HAO HU
### Project 3

train <- function(features,labels,pca,np,w){
  # pca <- prcomp(features)
  loadings <- as.data.frame(pca$x)
  modSVM <- svm(label_train~.,
                data=loadings[1:np],
               type='C-classification',
               kernel='linear',
               class.weights = w)
  model=modSVM
  return(model)
}