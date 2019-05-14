library(data.table)
library(rectools) 
library(recosystem)

source('nmf_CV.R')

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
#   - @time_hex: string, default '5cda01a9'
#        Time in hexidecimal for the .RDS to be used.
#
main <- function(csvs=csvs, testing300='data/jester-data-testing.csv', 
  proportions=c(0.3, 0.6, 0.9), ranks=c(1,seq(10, 60, by=10)), useRDS=T, 
  time_hex='5cda01a9', verbose=T)
{
  df <- dataProcessing(csvs)
  df_na <- df[is.na(rating)] # saves NAs for now
  df <- df[!is.na(rating)] 
  
  # shift rating to the positive direction by minimum rating
  shiftRatings <- function(x) {x + 10}
  df$rating <- shiftRatings(df$rating)

  df_300 <- fread(testing300)
  mtx_300 <- as.matrix(df_300[, 2:ncol(df_300)])
  mtx_300 <- shiftRatings(mtx_300)
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
                           
  # List of vectors to store training MAEs:
  #
  # mae_tr
  #   |
  #   |____ [['0.3']]
  #   |          |
  #   |          |____ [[strRnk1]] 
  #   |          |        |          [1]             [2]
  #   |          |        |____ c(training_MAE, training_MAE... )
  #   |          |
  #   |          |____ [[strRnk2]] 
  #   |          |        |          [1]             [2]
  #   |          |        |____ c(training_MAE, training_MAE... )
  #  ...        ...
  mae_tr <- setNames(vector('list', length(proportions)), as.character(proportions)) 
                     
  # Matrix to store testing mean absolute error for each combination:
  #     rnk1 rnk2 rnk3 rnk4 ...        
  # p1   x    x    x    x   ...        
  # p2   x    x    x    x   ...     
  # ...
  mae_est <- matrix(nrow=length(proportions), ncol=length(ranks))
  
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
    mae_tr[[strP]] = setNames(vector('list', length(ranks)), as.character(ranks))
    
    for(r in ranks)
    {
      if(verbose) cat(paste0('    Starting rank ', r, '\n'))
      
      # Allocate datatable
      ests_total <- data.table(matrix(nrow=300*100, ncol=3))
      colnames(ests_total) <- c('uID', 'jID', 'rating')
      curRow <- 1
      
      # Allocate vector for training MAEs
      strRnk <- as.character(r)
      mae_tr[[strP]][[strRnk]] <- double(length(trainSets))
      
      
      # For each pair of training/testing set. predict and save the results
      for(i in 1:length(trainSets))
      {
        if(verbose) cat(paste0('        Starting pair ', i, '... '))
        trainSet <- trainSets[[i]]
        testSet <- testSets[[i]]
        
        # Training/testing
        # 'capture.output' hides the output from 'trainReco'
        capture.output(trainedModel <- trainReco(trainSet[,-4], rnk = r, nmf = TRUE)) # training using trainSet
        estimates <- predict.RecoS3(trainedModel, testSet[, -(3:4)]) #predicting for testSet using trainedModel
        # Prediction on training set
        training_ests <- predict.RecoS3(trainedModel, trainSet[, -(3:4)]) 
        
        training_MAE <- mean(abs(training_ests - trainSet$rating))
        mae_tr[[strP]][[strRnk]][i] <- training_MAE
        
        testing_MAE <- mean(abs(estimates - testSet$rating))
        if(verbose) cat(paste0('Training MAE: ', training_MAE, 
                               '. Testing MAE: ', testing_MAE, '\n'))
        
        # Add to the pre-allocated space 
        ests_total$uID[curRow:(curRow + nrow(testSet) - 1)] <- testSet$uID
        ests_total$jID[curRow:(curRow + nrow(testSet) - 1)] <- testSet$jID
        ests_total$rating[curRow:(curRow + nrow(testSet) - 1)] <- estimates
        
        curRow <- curRow + nrow(testSet) 
      }
      
      # Sort them in order for easier extraction.
      ests_total <- ests_total[order(uID, jID)] 
      
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
      estimate_list[[strP]][[strRnk]] <- final_matrix
      # Store MAE for the current rank in the matrix
      rIdx <- which(p == proportions)
      cIdx <- which(r == ranks)
      mae_est[rIdx, cIdx] <- mean(abs(mtx_300 - final_matrix))
    }
  }
  
  out <- list(estimate_list, mae_tr, mae_est)
  
  # Save the output as .rds file
  time <- Sys.time() # for naming RDS files to avoid overwrite
  time_hex <- as.character(as.hexmode(as.integer(time)))
  strPs <- paste(p*100, collapse='_')
  strRnks <- paste(min(ranks), max(ranks), sep='_')
  saveRDS(out, paste0('./', dn, '/nmf_main_out_', strPs, '_', strRnks, '_', time_hex, '.rds'))
  
  out
}
