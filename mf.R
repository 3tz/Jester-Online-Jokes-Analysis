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


# Arguments: 
#   - @csvs: list, default @csvs 
#        A list of paths to the CSV files containing the datasets
#   - @nUser: integer default 6000
#        Number of users to be randomly chosen from the dataset as the test set
#   - @nJoke: integer default 5
#        Number of jokes to be chosen for each randomly chosen user
#   - @pTotal: vector of doubles
#        A vector of portions of the total dataset to be used. 0 < p < 1
run <- function(csvs=csvs, nUser=6000, nJoke=5, pTotal=c(0.1, 0.5, 1))
{
  df <- dataProcessing(csvs)
  df_na <- df[is.na(rating)] # saves NAs for now
  df <- df[!is.na(rating)] 

  # Create a test set with 6k random users and 5 random joke ratings
  testuID <- sample(max(df$uID), nUser) 
  df_temp <- df
  df_temp <- df_temp[, rowID := .I]
  df_temp <- df_temp[uID %in% testuID]
  
  
  testSet <- data.table(matrix(nrow=0, ncol=5)) # the result table
  colnames(testSet) <- c('uID', 'jID', 'rating', 'nRated', 'rowID')
  for(i in testuID)
  {
    testSet <- rbindlist(list(testSet, df_temp[uID == i][sample(max(nRated), nJoke)]))
  }
  
  df <- df[-testSet$rowID, 1:4] # remove all test set rows from train set
  testSet <- testSet[order(uID, jID, rating, nRated), 1:4]
  
    
  # Use only p * 100% of the whole dataset
  for(p in pTotal)
  {
    cat('Using ', p*100, '% of the total dataset\n', sep='')
    
    trainSet <- data.table(matrix(nrow=0, ncol=4))
    colnames(trainSet) <- c('uID', 'jID', 'rating', 'nRated') 

    # for each user
    for(i in 1:max(df$uID))
    {
      if(i %% 100 == 0)
        print(i)
      df_i <- df[uID == i]
      trainSet <- rbindlist(list(trainSet, df_i[sample(nrow(df_i), round(nrow(df_i) * p) )]))
    }
    
    # TODO: add NMF computation with @trainset
    
    #idx <-  sample(nrow(df), round(pTotal*nrow(df)))
    #df <- df[idx, ]
  }
    
  
  
}