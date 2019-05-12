library(data.table)
library(rectools) 
library(recosystem)

set.seed(9999)

csvs <- list('data/jester-data-1.csv', 
             'data/jester-data-2.csv', 
             'data/jester-data-3.csv')

# Convert the datasets into the following format: 
#            uID jID rating nRated
#       1:     1   1  -7.82     74
#       2:     1   2   8.79     74
#       3:     1   3  -9.66     74
#       4:     1   4  -8.16     74
#       5:     1   5  -7.52     74
#      ---                        
# 7342096: 73421  96     NA     35
# 7342097: 73421  97     NA     35
# 7342098: 73421  98     NA     35
# 7342099: 73421  99     NA     35
# 7342100: 73421 100     NA     35
# 
# Arguments: 
#   - @lPath2Csvs: list, default @csvs 
#        A list of paths to the CSV files containing the datasets
#   - @repNA: logical, default TRUE
#        Logical indicator for replacing rating of 99 with NA.
#
# Returns:
#   - A data.table with the 3-column + covariates format
dataProcessing <- function(lPath2Csvs=csvs, repNA=T)
{
  df <- rbindlist(lapply(csvs, fread))
  nRated <- df$V1 # number of jokes rated for each user
  df <- df[, 2:ncol(df)] # take out the part with only joke ratings
  
  rst <- data.table(matrix(nrow=0, ncol=4)) # the result table
  colnames(rst) <- c('uID', 'jID', 'rating', 'nRated')
  
  for(jID in 1:ncol(df))
  {
    rst <- rbindlist(list(rst, data.table(1:nrow(df), jID, df[[jID]], nRated)))
  }
  
  if(repNA)
    rst[rating==99]$rating <- NA
    
  rst[order(uID, jID, rating, nRated)]
}

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
      if(i %% 1000 == 0 && verbose) 
        cat(paste0('Current iteration: ', i, '\n'))
        
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
        # REMOVED: testSet[[2]] will just be trainSet[[1]] where uIDs in the 300
        # # Only include the rows that weren't in the first testing set
        # testSet[[2]] <- rbind(testSet[[2]], df_i[rows]) 
      }
      
    }
    testSet[[2]] <- rbind(testSet[[2]], trainSet[[1]][uID %in% testuIDs])    
  }
  else
  {
    # For each user ID, compute the p*100% joke IDs from the total jokes
    for(i in 1:max(df$uID))
    {
      if(i %% 1000 == 0 && verbose) 
        cat(paste0('Current iteration: ', i, '\n'))
        
      df_i <- df[uID == i]
      
      nRated <- df_i[1]$nRated # total number of jokes rated by user i
      idxs <- 1:nRated
      # Rounding
      tmp <- round((1-p)*100*nRated + 50 , 2)
      ntest <- floor(tmp/100)
      
      # ntest <- round((1-p) * nRated) # Number of jokes to be used for testing
      
      # split into training and testing     
      perm <- sample(nRated, nRated) 
      # List of test indices for each testing set
      l_testIdxs <- split(perm, perm %% nSets)
      
      for(j in 1:length(l_testIdxs))
      {
        testRows <- l_testIdxs[[j]]
        # cat(paste(nRated, ntest, nRated - ntest, length(idxs[!idxs %in% testRows]), '\n'))
        # print(sort(testRows))
        trainRows <- sample(idxs[!idxs %in% testRows], nRated-ntest)
        trainSet[[j]] <- rbind(trainSet[[j]], df_i[trainRows])
        
        # Add to testing set if it is one of the 300 users
        if(i %in% testuIDs)
        {
          testSet[[j]] <- rbind(testSet[[j]], df_i[testRows])
        }
      }
    }    
  }
  
  list(trainSet, testSet)
}

# Main function to run NMF on given datasets and produce output CSV files.
#
# Arguments: 
#   - @csvs: list, default @csvs 
#        A list of paths to the CSV files containing the complete datasets
#
#   - @testing300: str, default 'data/jester-data-testing.csv'
#       Path to the testing set with the chosen 300 users.
#
#   - @proportions: vector of doubles, default c(0.1, 0.5, 1)
#        A vector of portions of the number of jokes per user to be used for 
#        predictions. 0 < p < 1
#
#   - @ranks: vector of integers, default c(1, 10, 20, 30, ..., 290, 300)
#        A vector of ranks to be run by the NMF.
#
main <- function(csvs=csvs, testing300='data/jester-data-testing.csv', 
  proportions=c(0.3, 0.6, 0.9), ranks=c(1,seq(10, 60, by=10)))
{
  df <- dataProcessing(csvs)
  df_na <- df[is.na(rating)] # saves NAs for now
  df <- df[!is.na(rating)] 
  
  # Min-max normalization over ratings
  normalize <- function(x) {(x - min(x)) / (max(x) - min(x))}
  df$rating <- normalize(df$rating)

  df_300 <- fread(testing300)
  testuIDs <- df_300$UserID + 1 # uIDs start from 0 in jester-data-testing.csv
  #to store all the estimates for each proportion and rank combination
  estimate_list <- list() 
  i <- 1
  for(p in proportions)
  {
    l <- CV(df, p, testuIDs, T)
    save(l, file=paste0('./RData/nmf_CV_out_', p*100))
    
    trainSet1 = l[[1]][order(uID, jID)]
    trainSet2 = l[[2]][order(uID, jID)]
    testSet1 = l[[3]][order(uID, jID)]
    testSet2 = l[[4]][order(uID, jID)]
        
    for(r in ranks)
    {
      # Training/testing
      trainedModel1 <- trainReco(trainSet1[,-4], rnk = r, nmf = TRUE) # training using trainSet1
      estimates1 <- predict.RecoS3(trainedModel1, testSet1[,-(3:4)]) #predicting for testSet1 using trainedModel1
      
      trainedModel2 <- trainReco(trainSet2[,-4], rnk = r, nmf = TRUE) # training using trainSet2
      estimates2 <- predict.RecoS3(trainedModel2, testSet2[,-(3:4)]) #predicting for testSet1 using trainedModel2
      
      estimatedSet1 <- testSet1
      estimatedSet1[,3] <- estimates1 # predicted ratings along with uID ,jID and nRated
      
      estimatedSet2 <- testSet2
      estimatedSet2[,3] <- estimates2 # predicted ratings along with uID ,jID and nRated
      
      ests_total <- rbind(estimatedSet1 , estimatedSet2)
      ests_total <- ests_total[order(uID, jID)] # combined estimated data in form ('uID', 'jID', 'rating', 'nRated') 
      estimate_list[[i]] <- ests_total # list to maintain all the calculated estimates
      i <- i+1
      # TODO: output formating 
      # make csv file of estimated ratings of number of users x number of jokes in form ('UserID', 'J1', 'J2', 'J3',...,'J100')
      # calculate absolute value of empirical value - estimated value (i.e. actual rating - predicted rating)
      #...
      
    }
  }
  estimate_list
}
