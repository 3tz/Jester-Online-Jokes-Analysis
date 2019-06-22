import pandas as pd
import numpy as np
import pickle
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # Set path
np.set_printoptions(linewidth=200, threshold=np.nan, suppress=True)

# Produce outputs with specified formats for each training size.
#
# 1. recommendations.csv: Top Recommended jokes.
# uID rec1 rec2 rec3 rec4 ... rec100
#  1  jID  jID  jID  jID  ... jID
#  5  jID  jID  jID  jID  ... jID
#  7  jID  jID  jID  jID  ... jID
# ...
#
# 2. AE_mlp.csv: Absolute Errors.
# uID rec1 rec2 rec3 rec4 ... rec100
#  1   x    x    x    x   ...  x
#  5   x    x    x    x   ...  x
#  7   x    x    x    x   ...  x
# ...
#
# 3. TCV_mlp.csv: Ternary Categorical Variables.
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
# 5. EST_mlp.csv: Estimates with NMF.
# uID rec1 rec2 rec3 rec4 ... rec100
#  1   x    x    x    x   ...   x
#  5   x    x    x    x   ...   x
#  7   x    x    x    x   ...   x
# ...
# 6. TRUE_mlp.csv: True values but sorted according to recommendations.csv
# uID rec1 rec2 rec3 rec4 ... rec100
#  1   x    x    x    x   ...   x
#  5   x    x    x    x   ...   x
#  7   x    x    x    x   ...   x
# ...

def main(p, n_epoch, ts, testing300 = '../data/jester-data-testing.csv'):
    mtx_true = pd.read_csv(testing300)
    testuIDs = np.sort(mtx_true.UserID.unique())
    testuIDs = testuIDs.reshape([1, 300])
    mtx_true = mtx_true.values[:, 1:]

    # Comparison files
    tavg = pd.read_csv('../data/compare_totalAVG.csv').values
    unif = pd.read_csv('../data/compare_uniform.csv').values
    uavg = pd.read_csv('../data/compare_userAVG.csv').values

    # Number of ratings in each pair
    n_ratings = {'30': np.matrix([21000, 9000]),
                 '60': np.matrix([9900, 10200, 9900]),
                 '90': np.matrix([2400, 2700, 2700, 2700, 2700, 2400,
                                  2400, 2400, 2400, 2400, 2400, 2400])}[str(p)]

    with open('pkl/ests_{}_{}_{}.pkl'.format(p, n_epoch, ts), 'rb') as f:
        ests = pickle.load(f)
    with open('pkl/mae_tr_{}_{}_{}.pkl'.format(p, n_epoch, ts), 'rb') as f:
        mae_tr = pickle.load(f)
    with open('pkl/mae_va_{}_{}_{}.pkl'.format(p, n_epoch, ts), 'rb') as f:
        mae_va = pickle.load(f)

    # Find optimal epoch
    epoch = np.matmul(n_ratings, mae_va).argmin()
    print('Optimal Epoch:', epoch)
    ests = ests[epoch]

    # Generate absolute error matrix
    def genAE(ests, true, cols):
        mtx = np.abs(ests - true)
        # order them the same way as recoms
        for i in range(300):
            ind = recoms.values[i, 1:101] - 1
            mtx[i, ] = mtx[i, ind]

        dt = pd.DataFrame(mtx)
        dt.insert(0, '_', testuIDs[0])
        dt.columns = cols

        return dt

    def genTCV(ae):
        tcv = ae.copy()
        tcv[tcv[tcv.columns[1:]] < 3] = -1
        tcv[(tcv[tcv.columns[1:]] >= 3) & (tcv[tcv.columns[1:]] < 6)] = -2
        tcv[tcv[tcv.columns[1:]] >= 6] = -3
        tcv.replace(-1, 'a', inplace=True)
        tcv.replace(-2, 'b', inplace=True)
        tcv.replace(-3, 'c', inplace=True)
        assert(np.all(tcv[tcv.columns[1:]].isin(['a','b','c']).all()))

        return tcv

    def safe_mkdir(dn):
        try:
            os.mkdir(dn)
        except FileExistsError:
            pass

    # Create dir and sub-dir for the outputs
    dn = 'output'
    safe_mkdir(dn)
    dn = '{}/{}_{}/'.format(dn, p, epoch)
    safe_mkdir(dn)

    cols = ['uID'] + ['rec' + str(i+1) for i in range(100)]

    # recommendations.csv
    # negative b/c argsort gives increasing order; add 1 b/c jID starts from 1
    recoms = np.argsort(-ests) + 1
    recoms = np.hstack((testuIDs.T, recoms))
    recoms = pd.DataFrame(recoms, columns=cols)
    recoms.to_csv(dn + 'recommendation.csv', sep=',', index=False)

    # AE_mlp.csv
    ae_mlp = genAE(ests, mtx_true, cols)
    ae_mlp.to_csv(dn + 'AE_mlp.csv', sep=',', index=False)

    # EST_mlp.csv
    # Negative b/c argsort gives increasing order; * (-1) to restore the signs
    est_mlp = np.sort(-ests) * (-1)
    est_mlp = pd.DataFrame(est_mlp)
    est_mlp.insert(0, '_', testuIDs[0])
    est_mlp.columns = cols
    est_mlp.to_csv(dn + 'EST_mlp.csv', sep=',', index=False)

    # TRUE_mlp.csv
    true_mlp = mtx_true
    # order them the same way as recoms
    for i in range(300):
        ind = recoms.values[i, 1:101] - 1
        true_mlp[i, ] = true_mlp[i, ind]
    true_mlp = pd.DataFrame(true_mlp)
    true_mlp.insert(0, '_', testuIDs[0])
    true_mlp.columns = cols
    true_mlp.to_csv(dn + 'TRUE_mlp.csv', sep=',', index=False)

    # TCV_mlp.csv
    tcv_mlp = genTCV(ae_mlp)
    tcv_mlp.to_csv(dn + 'TCV_mlp.csv', sep=',', index=False)

    # AE_unif.csv
    ae_unif = genAE(unif[:, 1:101], mtx_true, cols)
    ae_unif.to_csv(dn + 'AE_unif.csv', sep=',', index=False)

    # AE_tavg.csv
    ae_tavg = genAE(tavg[:, 1:101], mtx_true, cols)
    ae_tavg.to_csv(dn + 'AE_tavg.csv', sep=',', index=False)

    # AE_uavg.csv
    ae_uavg = genAE(uavg[:, 1:101], mtx_true, cols)
    ae_uavg.to_csv(dn + 'AE_uavg.csv', sep=',', index=False)

    # TCV_unif.csv
    tcv_unif = genTCV(ae_unif)
    tcv_unif.to_csv(dn + 'TCV_unif.csv', sep=',', index=False)

    # TCV_tavg.csv
    tcv_tavg = genTCV(ae_tavg)
    tcv_tavg.to_csv(dn + 'TCV_tavg.csv', sep=',', index=False)

    # TCV_uavg.csv
    tcv_uavg = genTCV(ae_uavg)
    tcv_uavg.to_csv(dn + 'TCV_uavg.csv', sep=',', index=False)

    def nABC(tcv):
        print('Ternary Count: a: {}\tb: {}\tc: {}'
                 .format(np.sum(np.sum(tcv[cols[1:]] == 'a')),
                         np.sum(np.sum(tcv[cols[1:]] == 'b')),
                         np.sum(np.sum(tcv[cols[1:]] == 'c'))))
    nABC(tcv_mlp)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Must follow the following format:')
        print('python3 train_mlp.py {30, 60, 90} '
              '{n_epoch} {timestamp, \"current\"}')
    else:
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

