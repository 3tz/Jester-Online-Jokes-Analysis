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
#   - @useRDS: logical, default T
#        Indicator for using saved .RDS files instead of computing again.
#
#   - @time_hex: string, default '5cd816a5'
#        Time in hexidecimal for the .RDS to be used.
#
main <- function(csvs=csvs, testing300='data/jester-data-testing.csv', 
  proportions=c(0.3, 0.6, 0.9), ranks=c(1,seq(10, 60, by=10)), useRDS=T, 
  time_hex='5cd816a5', verbose=T)
{
  df <- dataProcessing(csvs)
  df_na <- df[is.na(rating)] # saves NAs for now
  df <- df[!is.na(rating)] 
  
  # Min-max normalization over ratings
  normalize <- function(x) {(x - min(x)) / (max(x) - min(x))}
  df$rating <- normalize(df$rating)

  df_300 <- fread(testing300)
  mtx_300 <- as.matrix(df_300[, 2:ncol(df_300)])
  mtx_300 <- normalize(mtx_300)
  testuIDs <- sort(df_300$UserID + 1) # uIDs start from 0 in jester-data-testing.csv
  
  # To store all the estimates for each proportion and rank combination in the 
  #   following format: 
  #
  # estimate_list
  #    |
  #    |____ [['0.3']]
  #    |          |
  #    |          |____ [[strRnk1]] 
  #    |          |        |____ final_matrix
  #    |          |           
  #    |          |____ [[strRnk2]] 
  #    |          |        |____ final_matrix
  #    |         ...
  #    |
  #    |____ [['0.6']]
  #    |          |
  #    |          |____ [[strRnk1]] 
  #    |          |        |____ final_matrix
  #    |          |           
  #    |          |____ [[strRnk2]] 
  #    |          |        |____ final_matrix
  #    |         ...
  #   ...
  estimate_list = setNames(vector('list', length(proportions)), as.character(proportions))
                           
  # Matrix to store mean absolute error for each combination:
  #     rnk1 rnk2 rnk3 rnk4 ...        
  # p1   x    x    x    x   ...        
  # p2   x    x    x    x   ...     
  # ...
  mae_mtx <- matrix(nrow=length(proportions), ncol=length(ranks))
  
  dn <- 'RData'
  if(!useRDS)
  {
    time <- Sys.time() # for naming RDS files to avoid overwrite
    time_hex <- as.character(as.hexmode(as.integer(time)))
    for(p in proportions)
    {
      l <- CV(df, p, testuIDs, T)
      saveRDS(l, paste0('./', dn, '/nmf_CV_out_', p*100, '_', time_hex, '.rds'))
    }
  }

  for(p in proportions)
  {
    if(verbose) cat(paste0('Starting training size ', p, '\n'))
    
    filename <- paste0('./', dn, '/nmf_CV_out_', p*100, '_', time_hex, '.rds')
    l <- readRDS(filename)
    trainSets <- l[[1]]
    testSets <- l[[2]]
    
    # List of estimates for all ranks
    strP = as.character(p)
    estimate_list[[strP]] = setNames(vector('list', length(ranks)), as.character(ranks))
    
    for(r in ranks)
    {
      if(verbose) cat(paste0('    Starting rank ', r, '\n'))
      
      # Allocate datatable
      ests_total <- data.table(matrix(nrow=300*100, ncol=3))
      colnames(ests_total) <- c('uID', 'jID', 'rating')
      curRow <- 1
      
      # For each pair of training/testing set. predict and save the results
      for(i in 1:length(trainSets))
      {
        if(verbose) cat(paste0('        Starting pair ', i, '\n'))
        trainSet <- trainSets[[i]]
        testSet <- testSets[[i]]
        
        # Training/testing
        # 'capture.output' hides the output from 'trainReco'
        capture.output(trainedModel <- trainReco(trainSet[,-4], rnk = r, nmf = TRUE)) # training using trainSet
        estimates <- predict.RecoS3(trainedModel, testSet[, -(3:4)]) #predicting for testSet using trainedModel
        
        # Add to the pre-allocated space 
        ests_total$uID[curRow:(curRow + nrow(testSet) - 1)] <- testSet$uID
        ests_total$jID[curRow:(curRow + nrow(testSet) - 1)] <- testSet$jID
        ests_total$rating[curRow:(curRow + nrow(testSet) - 1)] <- estimates
        
        curRow <- curRow + nrow(testSet) 
      }
      
      # output formating 
      # make matrix file of estimated ratings of number of users x number of 
      #   jokes in form:
      #            ('J1', 'J2', 'J3',...,'J100')
      # testuID1     x     x     x   ...   x          
      # testuID2     x     x     x   ...   x     
      #   ...        x     x     x   ...   x
      # testuID300   x     x     x   ...   x    
      
      # Allocate a (300 users) x (100 jokes) matrix
      final_matrix <- matrix(, nrow = 300, ncol = 100) 
      
      # testuIDs is strictly increasing, so just insert in increasing order
      for(i in 1:300){
        id <- testuIDs[i]
        final_matrix[i, ] <- ests_total$rating[which(ests_total$uID == id)]
      }
      
      # Store the results for current rank in estimate_list
      strRnk <- as.character(r)
      estimate_list[[strP]][[strRnk]] <- final_matrix
      # Store MAE for the current rank in mae_mtx
      rIdx <- which(p == proportions)
      cIdx <- which(r == ranks)
      mae_mtx[rIdx, cIdx] <- mean(abs(mtx_300 - final_matrix))
    }
  }
  
  out <- list(estimate_list, mae_mtx)
  
  # Save the output as .rds file
  time <- Sys.time() # for naming RDS files to avoid overwrite
  time_hex <- as.character(as.hexmode(as.integer(time)))
  strPs <- paste(p*100, collapse='_')
  strRnks <- paste(min(ranks), max(ranks), sep='_')
  saveRDS(out, paste0('./', dn, '/nmf_main_out_', strPs, '_', strRnks, '_', time_hex, '.rds'))
  
  out
}
