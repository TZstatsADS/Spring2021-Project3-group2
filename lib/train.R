###########################################################
### Train a classification model with training features ###
###########################################################

# Baseline gbm
# https://www.datatechnotes.com/2019/06/gradient-boosting-regression-example.html
# http://uc-r.github.io/gbm_regression#:~:text=gbm%20The%20gbm%20R%20package%20is%20an%20implementation,A%20presentation%20is%20available%20here%20by%20Mark%20Landry.
train_gbm <- function(train, cv = 5, n = 200, lr = 0.01){
    model = gbm(label ~.,
              data = train,
              distribution = "bernoulli",
              cv.folds = cv,
              shrinkage = lr, # learning rate
              n.minobsinnode = 10, # minimum number of observations in trees' terminal nodes.
              n.trees = n)
    return(model)
}

# logistic
train_logit <- function(features, labels, w = NULL, l = 1){
  model <- glmnet(features, labels, weights = w, alpha = 1, family = "binomial", lambda = l)
  return(model)
}
