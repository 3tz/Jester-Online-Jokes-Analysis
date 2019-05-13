library(data.table)

# Testing functino for nmf.R:CV(). "OK" will be printed if all test cases 
#   passed.
#
# Arguments: 
#   - @CV_out: str
#        String of the path to the .rds data file that contains the CV() output
#
#   - @p: double
#       Portions of the number of jokes per user used for CV(). 
#       0 < p < 1
#
#   - @testing300: str, default 'data/jester-data-testing.csv'
#       Path to the testing set with the chosen 300 users.
#
# Returns: NA
test_CV <- function(CV_out, p, testing300='data/jester-data-testing.csv')
{
  l <- readRDS(CV_out)
  train <- l[[1]]
  test <- l[[2]]
  
  
  df_300 <- fread(testing300)
  testuIDs <- df_300$UserID + 1 # uIDs start from 0 in jester-data-testing.csv
  
  curUser <- 0
  for(u in testuIDs)
  {
    combined <- data.table(matrix(nrow=0, ncol=4))
    colnames(combined) <- c('uID', 'jID', 'rating', 'nRated') 
    
    for(i in 1:length(test))
    {
      # Testset must NOT be in trainset
      stopifnot(!(any(test[[i]][uID == u]$jID %in% train[[i]][uID == u]$jID)))
      # Train size must be p * nRated
      stopifnot(nrow(train[[i]][uID == u]) == round(p*train[[i]][uID == u]$nRated[1]))
      combined <- rbind(combined, test[[i]][uID == u])
    }
    # combined testset should cover all jIDs
    stopifnot(nrow(combined$jID) == 100)
    stopifnot(all(sort(unique(combined$jID)) == 1:100))
    
    curUser <- curUser + 1
    
    if(curUser %% 100 == 0)
      cat(paste0(curUser, ' test users done.\n'))
  }
  total <- 0
  
  alluIDs <- 1:73421
  alljIDs <- 1:100
  
  for(df in train)
  {
    # Each training subset should contain all unique uIDs and jIDs
    stopifnot(all(sort(unique(df$uID)) == alluIDs))
    stopifnot(all(sort(unique(df$jID)) == alljIDs))
  }
  cat('OK\n')
}