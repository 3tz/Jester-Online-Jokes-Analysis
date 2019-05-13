# Produce pairs of training and testing sets to test on @testuIDs.
#
# Arguments:
#   - @p: double
#       Proportion of the number of jokes per user to be used for predictions.
#   
#   - @df: data.table
#       3-col data table that contains the whole dataset.
#
#   - @testuIDs: vector
#       A vector that contains all user IDs in the testing set.
#
#   - @verbose: logical, default FALSE
#       Indicator for showing iteration number.
#
#   - @seed: integer, default 9999
#       Seed for RNG. Set to NA to avoid reproducibility.
#
# Returns: list
#   - A list that contains the training and testing sets
CV <- function(df, p, testuIDs, verbose=F, seed=9999)
{
  if(!is.na(seed))
    set.seed(seed)
    
  # Create empty training and testing sets
  trainSet <- list()
  testSet <- list()
  
  # Number of sets
  # nSets <- ceiling(max(2, 1 / (1-p)))
  
  # Minimum of pairs, so there'll be at least round(p*nRated) number of 
  #   training data to be picked from for each user.
  tmp <- setNames(c(2, 3, 12), c('0.3', '0.6', '0.9'))
  nSets <- tmp[as.character(p)]
  
  if(is.na(nSets)) stop("p must be 0.3, 0.6, or 0.9")
        
  for(i in 1:nSets)
  {
    trainSet[[i]] <- data.table(matrix(nrow=0, ncol=4))
    testSet[[i]] <- data.table(matrix(nrow=0, ncol=4))
    colnames(trainSet[[i]]) <- c('uID', 'jID', 'rating', 'nRated') 
    colnames(testSet[[i]]) <- c('uID', 'jID', 'rating', 'nRated') 
  }
  
  if(verbose)
    cat(paste0('Total # of Users: ', max(df$uID),'\n'))
  
  if(p < 0.5)
  {
    # For each user ID, compute the p*100% joke IDs from the total jokes
    for(i in 1:max(df$uID))
    {
      if(i %% 1000 == 0 && verbose) cat(paste0('  Starting User: ', i, '\n'))
      
      df_i <- df[uID == i]
      
      nRated <- df_i[1]$nRated # total number of jokes rated by user i
      nTrain <- round(p * nRated) # Number of jokes to be used for training
          
      rows <- sample(nRated, nTrain) # row IDs for training    
      trainSet[[1]] <- rbind(trainSet[[1]], df_i[rows])
      # Pick row IDs from testing set 1 to gaurantee mut-ex
      rows2 <- sample((1:nRated)[-rows], nTrain)
      trainSet[[2]] <- rbind(trainSet[[2]], df_i[rows2])

      # Add to testing set if it is one of the 300 users
      if(i %in% testuIDs)
      {
        testSet[[1]] <- rbind(testSet[[1]], df_i[-rows])
      }
      
    }
    testSet[[2]] <- rbind(testSet[[2]], trainSet[[1]][uID %in% testuIDs])    
  }
  else
  {
    # For each user ID, compute the p*100% joke IDs from the total jokes
    for(i in 1:max(df$uID))
    {
      if(i %% 1000 == 0 && verbose) cat(paste0('  Starting User: ', i, '\n'))
        
      df_i <- df[uID == i]
      
      nRated <- df_i[1]$nRated # total number of jokes rated by user i
      # Rounding
      tmp <- round((1-p)*100*nRated + 50 , 2)
      ntest <- floor(tmp/100) # Number of jokes to be used for testing
      
      
      if(i %in% testuIDs) # Mutual-exclusion for users in testing300
      {
        # Find random permutation of the rated jIDs of this user   
        perm <- sample(nRated, nRated) 
        # Evenly plit the random permutation into nSets groups
        l_testIdxs <- split(perm, perm %% nSets)
        
        # For each of these groups, they will be the IDs in the testset
        for(j in 1:nSets)
        {
          testRows <- l_testIdxs[[j]]
          idxs <- 1:nRated
          trainRows <- sample(idxs[!idxs %in% testRows], nRated-ntest)
          trainSet[[j]] <- rbind(trainSet[[j]], df_i[trainRows])
          
          # Add to testing set if it is one of the 300 users
          testSet[[j]] <- rbind(testSet[[j]], df_i[testRows])
        }
      }
      else  # Just random sampling for other users
      {
        for(j in 1:nSets)
        {
          nTrain <- nRated - ntest
          trainRows <- sample(nRated, nTrain)
          trainSet[[j]] <- rbind(trainSet[[j]], df_i[trainRows])
        }
      }
    }    
  }
  
  # Sort by uID and jID
  for(i in 1:nSets)
  {
    trainSet[[i]] <- trainSet[[i]][order(uID,jID)]
    testSet[[i]] <- testSet[[i]][order(uID,jID)]
  }
  
  list(trainSet, testSet)
}