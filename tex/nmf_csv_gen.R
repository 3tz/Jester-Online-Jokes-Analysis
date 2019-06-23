library('data.table')

out <- readRDS('../nmf/RData/nmf_main_out_30_60_90_1_100_5cda01a9.rds')
estimate_list <- out[[1]]
mae_tr <- out[[2]]
mae_va <- out[[3]]

#          rnk1 rank2 ...
# pair1     x     x
# pair2     x     x
#  ...
mae_tr30 <- colMeans(matrix(unlist(mae_tr[['0.3']]), ncol=11))
mae_tr60 <- colMeans(matrix(unlist(mae_tr[['0.6']]), ncol=11))
mae_tr90 <- colMeans(matrix(unlist(mae_tr[['0.9']]), ncol=11))

#        rnk1 rank2 ...
# 0.3     x     x
# 0.6     x     x
# ...
mae_tr <- rbind(mae_tr30, mae_tr60, mae_tr90)
colnames(mae_tr) <- as.numeric(names(out[[2]][['0.3']]))
colnames(mae_va) <- as.numeric(names(out[[2]][['0.3']]))

write.table(mae_tr, file=paste0('csv/nmf_mae_tr.csv'), sep=',', row.names=F)
write.table(mae_va, file=paste0('csv/nmf_mae_va.csv'), sep=',', row.names=F)

out150 <- readRDS('../nmf/RData/nmf_main_out_90_150_150_5ce91bc9.rds')
out200 <- readRDS('../nmf/RData/nmf_main_out_90_200_200_5ce91dd9.rds')
out250 <- readRDS('../nmf/RData/nmf_main_out_90_250_250_5ce91fb9.rds')
out300 <- readRDS('../nmf/RData/nmf_main_out_90_300_300_5ce942e0.rds')
out350 <- readRDS('../nmf/RData/nmf_main_out_90_350_350_5ce9453f.rds')


 extra_mae_tr <- matrix(
                   c(mean(out150[[2]][[1]][[1]]), mean(out200[[2]][[1]][[1]]),
                     mean(out250[[2]][[1]][[1]]), mean(out300[[2]][[1]][[1]]),
                     mean(out350[[2]][[1]][[1]])), ncol=5)

extra_mae_va <- cbind(out150[[3]], out200[[3]], out250[[3]],
                      out300[[3]], out350[[3]])

colnames(extra_mae_va) <- c(150, 200, 250, 300, 350)
colnames(extra_mae_tr) <- c(150, 200, 250, 300, 350)

# extra for portion of 90%
write.table(extra_mae_tr, file=paste0('csv/nmf_mae_tr_extra_90.csv'), sep=',', row.names=F)
write.table(extra_mae_va, file=paste0('csv/nmf_mae_va_extra_90.csv'), sep=',', row.names=F)
