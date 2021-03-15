#############################################################
### Improve features by using Principal Components Analysis ###
#############################################################

feature_improved <- function(data, index = NULL){
  # By using Principal Components Analysis to improve our feature
  pca_res <- prcomp(data, scale = TRUE)
  if(!is.null(index)){return (pca_res$x[, 1:index])}
  # Feature select
  props <- pca_res$sdev^2/sum(pca_res$sdev^2)
  cumvar <- cumsum(props)
  # We want to select index is bigger than 0.9
  pc.index <- min(which(cumvar > 0.9))
  pc.index
  # Select data 
  data_selected <- pca_res$x[, 1:pc.index]
  return(data_selected)
}
