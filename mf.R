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
    rst[rating==99]$rating = NA
    
  rst[order(uID, jID, rating, nRated)]
}