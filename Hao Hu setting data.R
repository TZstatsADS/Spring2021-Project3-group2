install.packages("BiocManager")
install.packages("R.matlab")
install.packages("readxl")
install.packages("dplyr")
install.packages("readxl")
install.packages("ggplot2")
install.packages("caret")
install.packages("glmnet")
install.packages("WeightedROC")

library(R.matlab)
library(readxl)
library(dplyr)
library(EBImage)
library(ggplot2)
library(caret)
library(glmnet)
library(WeightedROC)

train_dir <- "./data/train_set/"
train_image_dir <- paste(train_dir, "images/", sep="")
train_pt_dir <- paste(train_dir, "points/", sep="")
train_label_path <- paste(train_dir, "label.csv", sep="")

run.cv <- TRUE 
sample.reweight <- TRUE 
K <- 5 
run.feature.train <- TRUE 
run.test <- TRUE 
run.feature.test <- TRUE 

lmbd = c(1e-3, 5e-3, 1e-2, 5e-2, 1e-1)
model_labels = paste("LASSO Penalty with lambda =", lmbd)

##import data
setwd("C:/Users/sluo1/Desktop/5243/other/Spring2021-Project3-group-2")
getwd()
info <- read.csv(train_label_path)
n <- nrow(info)
n_train <- round(n*(4/5), 0)
train_idx <- sample(info$Index, n_train, replace = F)
test_idx <- setdiff(info$Index, train_idx)

#store image
n_files <- length(list.files(train_image_dir))
image_list <- list()
for(i in 1:100){
    image_list[[i]] <- readImage(paste0(train_image_dir, sprintf("%04d", i), ".jpg"))
}

#store Fiducial points
#function to read fiducial points--input: index; output: matrix of fiducial points corresponding to the index
readMat.matrix <- function(index){
    return(round(readMat(paste0(train_pt_dir, sprintf("%04d", index), ".mat"))[[1]],0))
}

#load fiducial points
fiducial_pt_list <- lapply(1:n_files, readMat.matrix)
save(fiducial_pt_list, file="./output/fiducial_pt_list.RData")


##------------------------------construct features and responses-----------
source("./lib/feature.R")
tm_feature_train <- NA
if(run.feature.train){
    tm_feature_train <- system.time(dat_train <- feature(fiducial_pt_list, train_idx))
    save(dat_train, file="./output/feature_train.RData")
}else{
    load(file="./output/feature_train.RData")
}
tm_feature_test <- NA
if(run.feature.test){
    tm_feature_test <- system.time(dat_test <- feature(fiducial_pt_list, test_idx))
    save(dat_test, file="./output/feature_test.RData")
}else{
    load(file="./output/feature_test.RData")}

##---------Train a classification model with training features and responses------

source("./lib/train.R")
source("./lib/test.R")

# Model selection with cross-validation
source("./lib/cross_validation.R")
feature_train = as.matrix(dat_train[, -6007])
label_train = as.integer(dat_train$label)
if(run.cv){
    res_cv <- matrix(0, nrow = length(lmbd), ncol = 4)
    for(i in 1:length(lmbd)){
        cat("lambda = ", lmbd[i], "\n")
        res_cv[i,] <- cv.function(features = feature_train, labels = label_train, K,
                                  l = lmbd[i], reweight = sample.reweight)
        save(res_cv, file="./output/res_cv.RData")
    }
}else{
    load("./output/res_cv.RData")
}

## Visualize cross-validation results.
res_cv <- as.data.frame(res_cv)
colnames(res_cv) <- c("mean_error", "sd_error", "mean_AUC", "sd_AUC")
res_cv$k = as.factor(lmbd)
if(run.cv){
    p1 <- res_cv %>%
        ggplot(aes(x = as.factor(lmbd), y = mean_error,
                   ymin = mean_error - sd_error, ymax = mean_error + sd_error)) +
        geom_crossbar() +
        theme(axis.text.x = element_text(angle = 90, hjust = 1))
    p2 <- res_cv %>%
        ggplot(aes(x = as.factor(lmbd), y = mean_AUC,
                   ymin = mean_AUC - sd_AUC, ymax = mean_AUC + sd_AUC)) +
        geom_crossbar() +
        theme(axis.text.x = element_text(angle = 90, hjust = 1))
    print(p1)
    print(p2)
    
}
par_best <- lmbd[which.min(res_cv$mean_error)] # lmbd[which.max(res_cv$mean_AUC)]


# training weights
weight_train <- rep(NA, length(label_train))
for (v in unique(label_train)){
    weight_train[label_train == v] = 0.5 * length(label_train) / length(label_train[label_train == v])
}
if (sample.reweight){
    tm_train <- system.time(fit_train <- train(feature_train, label_train, w = weight_train, par_best))
} else {
    tm_train <- system.time(fit_train <- train(feature_train, label_train, w = NULL, par_best))
}
save(fit_train, file="./output/fit_train.RData")


##-------------Run test on test images-------------------------
tm_test = NA
feature_test <- as.matrix(dat_test[, -6007])
if(run.test){
    load(file="./output/fit_train.RData")
    tm_test <- system.time({label_pred <- as.integer(test(fit_train, feature_test, pred.type = 'class'));
                            prob_pred <- test(fit_train, feature_test, pred.type = 'response')})
}

## reweight the test data to represent a balanced label distribution
label_test <- as.integer(dat_test$label)
weight_test <- rep(NA, length(label_test))
for (v in unique(label_test)){
    weight_test[label_test == v] = 0.5 * length(label_test) / length(label_test[label_test == v])
}
accu <- sum(weight_test * (label_pred == label_test)) / sum(weight_test)
tpr.fpr <- WeightedROC(prob_pred, label_test, weight_test)
auc <- WeightedAUC(tpr.fpr)

## Summarize Running Time---------------

cat("The accuracy of model:", model_labels[which.min(res_cv$mean_error)], "is", accu*100, "%.\n")
cat("The AUC of model:", model_labels[which.min(res_cv$mean_error)], "is", auc, ".\n")

cat("Time for training model=", tm_train[1], "s \n")
cat("Time for testing model=", tm_test[1], "s \n")

