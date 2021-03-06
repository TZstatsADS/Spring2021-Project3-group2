#############################################################
### Construct features and responses for training images  ###
#############################################################
library(spatstat)

## =================================================================
## Feature set 0: pairwise distances between x and between y =======
## =================================================================

feature <- function(input_list = fiducial_pt_list, index){
  
  ### Input: a list of images or fiducial points; index: train index or test index
  ### Output: a data frame containing: features and a column of label
  
  ### Step 1: calculate pairwise distance of items in a vector
  pairwise_dist <- function(vec){
    ### input: a vector(length n), output: a vector containing pairwise distances(length n(n-1)/2)
    return(as.vector(dist(vec)))
  }
  
  ### Step 2: apply function in Step 1 to column of a matrix 
  pairwise_dist_result <-function(mat){
    ### input: a n*2 matrix(e.g. fiducial_pt_list[[1]]), output: a vector(length n(n-1))
    return(as.vector(apply(mat, 2, pairwise_dist))) 
  }
  
  ### Step 3: apply function in Step 2 to selected index of input list, output: a feature matrix with ncol = n(n-1) = 78*77 = 6006
  pairwise_dist_feature <- t(sapply(input_list[index], pairwise_dist_result))
  dim(pairwise_dist_feature) 
  
  ### Step 4: construct a dataframe containing features and label with nrow = length of index
  ### column bind feature matrix in Step 3 and corresponding features
  pairwise_data <- cbind(pairwise_dist_feature, info$label[index])
  ### add column names
  colnames(pairwise_data) <- c(paste("feature", 1:(ncol(pairwise_data)-1), sep = ""), "label")
  ### convert matrix to data frame
  pairwise_data <- as.data.frame(pairwise_data)
  ### convert label column to factor
  pairwise_data$label <- as.factor(pairwise_data$label)
  
  return(feature_df = pairwise_data)
}


## =================================================================
## Feature set 1: pairwise distances between all points ============
## =================================================================


# Function:  pairwise distances between fiducial points for samples with index in idx
GetPairwiseFeature <- function(pt_list, idx){
  
  ### Input: pt_list -> a list of images or fiducial points; idx -> train index or test index
  ### Output: a data frame containing: features and a column of label of binary class (-1,1)
  
  
  # Step 1: calculate pairwise distances between points
  Pdist <- function(mat){
    ### input: a 78*2 matrix with columns being X and Y coordinates of the points
    ### output: a 78*78 matrix of all pairwise spatial distances
    
    pts_df <- data.frame(mat)
    # calculate pairwise spatial distances
    pair_dist <- spatstat::crossdist(pts_df$X1, pts_df$X2,
                                     pts_df$X1, pts_df$X2)
    return(pair_dist)
  }
  
  # Step 2:  convert the matrix to a vector and remove duplicate pairs
  Pdist_vec <- function(mat){
    ### input: a 78*2 matrix with columns being X and Y coordinates of the points
    ### output: a 78*(78-1)/2 vector of all unique pairwise spatial distances
    
    pair_dist <- Pdist(mat)
    pair_dist_l <- pair_dist[lower.tri(pair_dist, diag = FALSE)] # Extract the lower half of the matrix to a vector
    return(pair_dist_l)
  }
  
  # Step 3: extract names of the pairs
  col_names <- colnames(data.frame(Pdist(pt_list[[1]]))) # get all point indexes
  pair_names <- outer(col_names, col_names, paste, sep="_") # paste the indexes pairwise to get names of the pairs
  pair_names_l <- pair_names[lower.tri(pair_names, diag = FALSE)]  # Only extract the lower half of the matrix to a vector
  
  # Step 4: apply the function in step 2 to idx of the input pt_list
  features_df1 <- t(sapply(pt_list[idx], Pdist_vec))
  dim(features_df1) 
  features_df1 <- as.data.frame(features_df1)
  colnames(features_df1) <- pair_names_l   # Change column names
  features_df1$labels <- info$label[idx] # Add outcome variable
  features_df1$labels[features_df1$labels == 0] <- -1 # Change class 0 to -1
  return(features_df1)
}


