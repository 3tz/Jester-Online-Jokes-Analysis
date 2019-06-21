import sys
import pickle
from mlp import *

class SaveResults(keras.callbacks.Callback):
    """
    Save results to estimates, mae_tr, and mae_va after each epoch.
    """
    def __init__(self, ests3, mae_tr, mae_va, cur_row, idx_pair, model):
        """
        Arguments:
            - ests3: list
                Three column version of `ests`
            - mae_tr: np.matrix
                Matrix of every pair's training MAE at each epoch.
            - mae_va: np.matrix
                Matrix of every pair's validation MAE at each epoch.
            - cur_row: int
                Current starting row index to insert new rows.
            - idx_pair: int
                Current pair index.
            - model: Mlp object
                Mlp object that is being used.
        """
        self.ests3 = ests3
        self.mae_tr = mae_tr
        self.mae_va = mae_va
        self.i = cur_row
        self.idx_pair = idx_pair
        self.m = model

    def on_epoch_end(self, epoch, logs=None):
        idx = self.idx_pair

        m = self.m
        # preds_tr = (m.model.predict([m.tr.uID, m.tr.jID])).flatten()
        preds_va = (m.model.predict([m.va.uID, m.va.jID])).flatten()

        # Should be all zeros before getting written on
        # assert(np.all(self.ests3[epoch][self.i:(self.i+m.va.shape[0]), :]
        #               == 0))

        # Insert to the pre-allocated matrix
        self.ests3[epoch][self.i:(self.i+m.va.shape[0]), 0] = m.va.uID
        self.ests3[epoch][self.i:(self.i+m.va.shape[0]), 1] = m.va.jID
        self.ests3[epoch][self.i:(self.i+m.va.shape[0]), 2] = preds_va

        # Training errors from predict() are supposed to be different from
        #   fit() due to training is done in batches.
        self.mae_tr[idx, epoch] = logs['mean_absolute_error']
        self.mae_va[idx, epoch] = logs['val_mean_absolute_error']

def main(p, n_epoch):
    """
    Arguments:
        - p: int
            Proportion of training set to run. Must be 30, 60, or 90.
        - n_epoch: int
            Number of epochs to train.
    """
    assert(p in [30, 60, 90])
    print('p: {}; n_epoch: {}'.format(p, n_epoch))
    n_pairs = {'30':2, '60':3, '90':12}[str(p)]

    # prefixes for the files
    pfs = [['cvout/30_' + str(i+1) for i in range(2)],
           ['cvout/60_' + str(i+1) for i in range(3)],
           ['cvout/90_' + str(i+1) for i in range(12)]]
    # Choose the corresponding prefixes
    pfs = pfs[int(p/30-1)]

    # List of estimates made by the model by combining all pairs results at
    #   each epoch.
    #
    #  ests
    #   |____ [epoch 1]
    #   |        |____ final_matrix
    #   |____ [epoch 2]
    #   |        |____ final_matrix
    #  ...
    #
    # where final_matrix has the following format
    #
    #            ('J1', 'J2', 'J3',...,'J100')
    # testuID1     x     x     x   ...   x
    #   ...        x     x     x   ...   x
    # testuID300   x     x     x   ...   x
    ests = []

    # Similar to `ests`; however, each element stores a 30K*3 matrix, and
    #   it will be used to form `ests`.
    ests3 = []

    for i in range(n_epoch):
        ests.append(np.zeros([300, 100]))
        ests3.append(np.zeros([300*100, 3]))

    # Matrix of every pair's training MAE at each epoch from Keras fit().
    #
    #         epoch1 epoch2 ...
    # pair1     x      x   ...
    # pair2     x      x   ...
    #  ...
    mae_tr = np.zeros([n_pairs, n_epoch])

    # Matrix of every pair's validation MAE at each epoch.
    #
    #         epoch1 epoch2 ...
    # pair1     x      x   ...
    # pair2     x      x   ...
    #  ...
    mae_va = np.zeros([n_pairs, n_epoch])

    cur_row = 0

    # for prefix of each pair in the set
    for idx, f in enumerate(pfs):
        print('########## Pair {} ##########'.format(idx))
        tr = pd.read_csv(f + '_train.csv')  # training
        va = pd.read_csv(f + '_test.csv')   # validation
        # unshift the rating done in NMF
        tr.rating -= 10
        va.rating -= 10

        m = Mlp(tr, va, lr=0.1, batch_size=4096,
                layer_sizes=[(16, 3), 200, 100], verbose=True, verbose_fit=1)
        m.new_model()
        m.train_model(n_epoch, callbacks=[SaveResults(ests3, mae_tr, mae_va,
                                                      cur_row, idx, m)])
        cur_row += va.shape[0]

    # Sorted unique test user IDs, so we can perform quick row # lookup.
    uIDs = np.sort(np.unique(ests3[0][:, 0]))

    for n in range(n_epoch):
        # assert(np.all(uIDs == np.sort(np.unique(ests3[n][:, 0]))))
        mtx3 = ests3[n] # 3-col matrix
        # convert to the format of `ests`
        for i in range(mtx3.shape[0]):
            (u, j, r) = mtx3[i, :] # uID, jID, rating
            u = np.where(uIDs == u)[0][0]
            j = int(j) - 1  # uIDs and jIDs start from 1
            ests[n][u, j] = r

    dn = 'pkl/'
    try:
        os.mkdir(dn)
    except FileExistsError:
        pass

    # Time-stamp for saving to avoid overwrite
    ts = hex(int((datetime.now()).timestamp()))[2:]

    # pkl/ests.pkl
    with open('{}ests_{}_{}_{}.pkl'.format(dn, p, n_epoch, ts), 'wb') as f:
        pickle.dump(ests, f)
    # pkl/mae_tr.pkl
    with open('{}mae_tr_{}_{}_{}.pkl'.format(dn, p, n_epoch, ts), 'wb') as f:
        pickle.dump(mae_tr, f)
    # pkl/mae_va.pkl
    with open('{}mae_va_{}_{}_{}.pkl'.format(dn, p, n_epoch, ts), 'wb') as f:
        pickle.dump(mae_va, f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Must follow the following format:')
        print('python3 train_mlp.py {30, 60, 90} {n_epoch}')
    else:
        main(int(sys.argv[1]), int(sys.argv[2]))
