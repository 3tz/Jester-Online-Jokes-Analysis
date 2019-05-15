library(data.table)

# Produce outputs with specified formats for each training size.
# 
# 1. recommendations.csv: Top Recommended jokes.    
# uID rec1 rec2 rec3 rec4 ... rec100
#  1  jID  jID  jID  jID  ... jID  
#  5  jID  jID  jID  jID  ... jID
#  7  jID  jID  jID  jID  ... jID 
# ...
# 
# 2. AE_nmf.csv: Absolute Errors.    
# uID rec1 rec2 rec3 rec4 ... rec100
#  1   x    x    x    x   ...  x 
#  5   x    x    x    x   ...  x 
#  7   x    x    x    x   ...  x 
# ...
#
# 3. TCV_nmf.csv: Ternary Categorical Variables.       
# uID rec1 rec2 rec3 rec4 ... rec100
#  1   x    x    x    x   ...  x 
#  5   x    x    x    x   ...  x 
#  7   x    x    x    x   ...  x 
# ...
#
# 4. AE_unif.csv: Absolute Errors against Uniformly Random.   
#    AE_tavg.csv: Absolute Errors against Total Average.   
#    AE_uavg.csv: Absolute Errors against User Average.
#    TCV_unif.csv: Ternary Cat Var for Uniformly Random.   
#    TCV_tavg.csv: Ternary Cat Var for Total Average.   
#    TCV_uavg.csv: Ternary Cat Var for User Average.
#
# 5. estimates.csv: Estimates with NMF.  
# uID rec1 rec2 rec3 rec4 ... rec100
#  1  jID  jID  jID  jID  ... jID  
#  5  jID  jID  jID  jID  ... jID
#  7  jID  jID  jID  jID  ... jID 
# ...
#  
# Arguments:
#   - @mainOut: str
#        String that contains the path to the .rds output of nmf::main().
#
#   - @proportions: vector of doubles, default c(0.3, 0.6, 0.9)
#        A vector of portions of the number of jokes per user to be used for 
#        predictions. 0 < p < 1
#
#   - @optimalRnks: str, default 'auto'; vector of integers
#        The ranks to use as the optimal rank for each . If 'auto', rank with the minimum
#        testing MAE will be used.
#
#   - @testing300: str, default 'data/jester-data-testing.csv'
#       Path to the testing set with the chosen 300 users.
#
# Returns: 
#   - NA
nmf_format <- function(mainOut, p=c(0.3, 0.6, 0.9), optimalRnks='auto', 
                       testing300='data/jester-data-testing.csv')
{
  shiftRatings <- function(x) {x + 10} 
  df_300 <- fread(testing300)
  mtx_true <- as.matrix(df_300[, 2:ncol(df_300)])
  mtx_true <- shiftRatings(mtx_true)
  testuIDs <- sort(df_300$UserID)  # No plus one to match the given CSVs.
  
  tavg <- as.matrix(read.csv('data/compare_totalAVG.csv'))
  unif <- as.matrix(read.csv('data/compare_uniform.csv'))
  uavg <- as.matrix(read.csv('data/compare_userAVG.csv'))
  
  out <- readRDS(mainOut)
  strPs <- as.character(p)
 
  l_est <- out[[1]] # estimate_list
  mae_est <- out[[3]] # Estimate MAE
  
  # Find optimal ranks
  if(all(optimalRnks == 'auto'))
  {
    optimalRnks <- integer(length(p))
    
    # For each sublist for p in l_est
    for(i in 1:length(p))
    {
      l <- l_est[[strPs[i]]]
      ranks <- names(l)
      optimalRnks[i] <- which.min(mae_est[i, ])
    }
  }
  cat('Optimal Ranks:', optimalRnks, '\n')
  # Generate AE matrix
  genAE <- function(ests, true)
  {
    mtx <- abs(ests - true)
    # order them the same way as recoms
    for(j in 1:300)
      mtx[j, ] <- mtx[j, recoms[j, 2:101]]
    mtx <- cbind(testuIDs, mtx)
    colnames(mtx) <- c('uID', paste0('rec', 1:100))
    mtx
  }
  
  genTCV <- function(ae)
  {
    tcv <- ae
    tcv[, 2:101][which(tcv[, 2:101] < 3)] <- -1
    tcv[, 2:101][which(tcv[, 2:101] >= 3 & tcv[, 2:101] < 6)] <- -2
    tcv[, 2:101][which(tcv[, 2:101] >= 6)] <- -3
    tcv[tcv == -1] <- 'a'
    tcv[tcv == -2] <- 'b' 
    tcv[tcv == -3] <- 'c' 
    
    tcv
  }
  
  # For each p
  for(i in 1:length(strPs))
  {
    dn <- paste0('./output/', p[i]*100, '/')
    
    jIDs <- 1:100
    rnk <- optimalRnks[i]
    ests <- l_est[[strPs[i]]][[rnk]]
    
    # recommendations.csv
    recoms <- t(apply(ests, 1, order, decreasing=T))
    recoms <- cbind(testuIDs, recoms)
    colnames(recoms) <- c('uID', paste0('rec', 1:100))
    write.table(recoms, file=paste0(dn, 'recommendations.csv'), row.names=F)
    
    # AE_nmf.csv
    ae_nmf <- genAE(ests, mtx_true)
    write.table(ae_nmf, file=paste0(dn, 'AE_nmf.csv'), row.names=F)
    
    # TCV_nmf.csv 
    tcv_nmf <- genTCV(ae_nmf)
    write.table(tcv_nmf, file=paste0(dn, 'tcv_nmf.csv'), row.names=F)
    # tcv_nmf <- ae_nmf
    # tcv_nmf[, 2:101][which(tcv_nmf[, 2:101] < 3)] <- -1
    # tcv_nmf[, 2:101][which(tcv_nmf[, 2:101] >= 3 & tcv_nmf[, 2:101] < 6)] <- -2
    # tcv_nmf[, 2:101][which(tcv_nmf[, 2:101] >= 6)] <- -3
    # tcv_nmf[tcv_nmf == -1] <- 'a'
    # tcv_nmf[tcv_nmf == -2] <- 'b' 
    # tcv_nmf[tcv_nmf == -3] <- 'c' 
    # write.table(tcv_nmf, file=paste0(dn, 'tcv_nmf.csv'), row.names=F)
    
    # AE_unif.csv
    ae_unif <- genAE(unif[, 2:101], mtx_true)
    write.table(ae_unif, file=paste0(dn, 'AE_unif.csv'), row.names=F)
    
    # AE_tavg.csv
    ae_tavg <- genAE(tavg[, 2:101], mtx_true)
    write.table(ae_tavg, file=paste0(dn, 'AE_tavg.csv'), row.names=F)
    
    # AE_uavg.csv
    ae_uavg <- genAE(uavg[, 2:101], mtx_true)
    write.table(ae_uavg, file=paste0(dn, 'AE_uavg.csv'), row.names=F)
    
    # TCV_unif.csv
    tcv_unif <- genTCV(ae_unif)
    write.table(tcv_unif, file=paste0(dn, 'tcv_unif.csv'), row.names=F)
    
    # TCV_tavg.csv
    tcv_tavg <- genTCV(ae_tavg)
    write.table(tcv_tavg, file=paste0(dn, 'tcv_tavg.csv'), row.names=F)
    
    # TCV_uavg.csv
    tcv_uavg <- genTCV(ae_uavg)
    write.table(tcv_uavg, file=paste0(dn, 'tcv_uavg.csv'), row.names=F)
  }

 
  
  
}