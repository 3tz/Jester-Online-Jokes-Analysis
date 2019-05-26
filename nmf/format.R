library(data.table)

# Change current directory to nmf/
dir <- getSrcDirectory(function(x) {})
setwd(dir)


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
# 5. EST_nmf.csv: Estimates with NMF.  
# uID rec1 rec2 rec3 rec4 ... rec100
#  1   x    x    x    x   ...   x  
#  5   x    x    x    x   ...   x
#  7   x    x    x    x   ...   x 
# ...
# 6. TRUE_nmf.csv: True values but sorted according to recommendations.csv
# uID rec1 rec2 rec3 rec4 ... rec100
#  1   x    x    x    x   ...   x 
#  5   x    x    x    x   ...   x
#  7   x    x    x    x   ...   x
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
#   - @testing300: str, default '../data/jester-data-testing.csv'
#       Path to the testing set with the chosen 300 users.
#
# Returns: 
#   - NA
nmf_format <- function(mainOut='./RData/nmf_main_out_30_60_90_1_100_5cda01a9.rds', 
                       p=c(0.3, 0.6, 0.9), optimalRnks='auto', 
                       testing300='../data/jester-data-testing.csv')
{
  shiftRatings <- function(x) {x + 10} 
  df_300 <- fread(testing300)
  mtx_true <- as.matrix(df_300[, 2:ncol(df_300)])
  mtx_true <- shiftRatings(mtx_true)
  testuIDs <- sort(df_300$UserID)  # No plus one to match the given CSVs.
  
  tavg <- as.matrix(read.csv('../data/compare_totalAVG.csv'))
  unif <- as.matrix(read.csv('../data/compare_uniform.csv'))
  uavg <- as.matrix(read.csv('../data/compare_userAVG.csv'))
  
  out <- readRDS(mainOut)
  strPs <- as.character(p)
 
  l_est <- out[[1]] # estimate_list
  mae_est <- out[[3]] # Estimate MAE
  ranks <- names(l_est[[1]]) # ranks
  # Find optimal ranks
  if(all(optimalRnks == 'auto'))
  {
    optimalRnks <- character(length(p))
    
    # For each sublist for p in l_est
    for(i in 1:length(p))
    {
      l <- l_est[[strPs[i]]]
      ridx <- which(names(l_est) == p[i])
      optimalRnks[i] <- ranks[which.min(mae_est[ridx, ])]
    }
  }
  else
    optimalRnks <- as.character(optimalRnks)
    
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
    dn <- paste0(p[i]*100, '_', optimalRnks[i])
    dn <- paste0('./output/', dn, '/')
    dir.create(dn, showWarnings=F)
    
    jIDs <- 1:100
    rnk <- optimalRnks[i]
    ests <- l_est[[strPs[i]]][[rnk]]
    
    # recommendations.csv
    recoms <- t(apply(ests, 1, order, decreasing=T))
    recoms <- cbind(testuIDs, recoms)
    colnames(recoms) <- c('uID', paste0('rec', 1:100))
    write.table(recoms, file=paste0(dn, 'recommendations.csv'), sep=',', row.names=F)
    
    # AE_nmf.csv
    ae_nmf <- genAE(ests, mtx_true)
    write.table(ae_nmf, file=paste0(dn, 'AE_nmf.csv'), sep=',', row.names=F)
    
    # EST_nmf.csv
    est_nmf <- ests
    est_nmf <- t(apply(est_nmf, 1, sort, decreasing=T))
    est_nmf <- cbind(testuIDs, est_nmf)
    colnames(est_nmf) <- c('uID', paste0('rec', 1:100))
    write.table(est_nmf, file=paste0(dn, 'EST_nmf.csv'), sep=',', row.names=F)
    
    # TRUE_nmf.csv
    true_nmf <- mtx_true
    # order them the same way as recoms
    for(j in 1:300)
      true_nmf[j, ] <- true_nmf[j, recoms[j, 2:101]]
      
    true_nmf <- cbind(testuIDs, true_nmf)
    colnames(true_nmf) <- c('uID', paste0('rec', 1:100))
    write.table(true_nmf, file=paste0(dn, 'TRUE_nmf.csv'), sep=',', row.names=F)    
    
    # TCV_nmf.csv 
    tcv_nmf <- genTCV(ae_nmf)
    write.table(tcv_nmf, file=paste0(dn, 'TCV_nmf.csv'), sep=',', row.names=F)

    # AE_unif.csv
    ae_unif <- genAE(unif[, 2:101], mtx_true)
    write.table(ae_unif, file=paste0(dn, 'AE_unif.csv'), sep=',', row.names=F)

    # AE_tavg.csv
    ae_tavg <- genAE(tavg[, 2:101], mtx_true)
    write.table(ae_tavg, file=paste0(dn, 'AE_tavg.csv'), sep=',', row.names=F)

    # AE_uavg.csv
    ae_uavg <- genAE(uavg[, 2:101], mtx_true)
    write.table(ae_uavg, file=paste0(dn, 'AE_uavg.csv'), sep=',', row.names=F)

    # TCV_unif.csv
    tcv_unif <- genTCV(ae_unif)
    write.table(tcv_unif, file=paste0(dn, 'TCV_unif.csv'), sep=',', row.names=F)

    # TCV_tavg.csv
    tcv_tavg <- genTCV(ae_tavg)
    write.table(tcv_tavg, file=paste0(dn, 'TCV_tavg.csv'), sep=',', row.names=F)

    # TCV_uavg.csv
    tcv_uavg <- genTCV(ae_uavg)
    write.table(tcv_uavg, file=paste0(dn, 'TCV_uavg.csv'), sep=',', row.names=F)
  }
  
}