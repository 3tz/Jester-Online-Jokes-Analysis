from mlp import *

class SaveResults(keras.callbacks.Callback):
    """
    Save results to estimates, mae_tr, and mae_va after each epoch.
    """
    def __init__(self, ests_total, cur_row, model):
        """
        Arguments:
            ests_total: list
                List to be modified.
            cur_row: int
                Current starting row index to insert new rows.
            model: Mlp object
                Mlp object that is being used.
        """
        self.ests_total = ests_total
        self.cur_row = cur_row
        self.m = model

    def on_epoch_end(self, epoch, logs=None):
        print('epoch {} ended.'.format(epoch))
        m = self.m
        estimates = m.model.predict([m.val.uID, m.val.jID])

        #self.ests_total[epoch][self.cur_row:(self.cur_row+m.va.shape[0]), 0]


def main(n_epoch=3):
    # prefixes for the files
    pfs = [['cvout/30_' + str(i+1) for i in range(2)],
           ['cvout/60_' + str(i+1) for i in range(3)],
           ['cvout/90_' + str(i+1) for i in range(12)]]

    # List of predictions made by the model by combining all pairs results at
    #   each epoch.
    #
    # estimates
    #     |____ [epoch 1]
    #     |        |____ final_matrix
    #     |____ [epoch 2]
    #     |        |____ final_matrix
    #    ...
    estimates = []

    #   List of every pair's MAE training error at each epoch.
    #
    #   mae_tr
    #     |____ [epoch 1]
    #     |        |        [pair 1]      [pair 2]
    #     |        |____ [training_MAE, training_MAE, ...]
    #     |____ [epoch 2]
    #     |        |        [pair 1]      [pair 2]
    #     |        |____ [training_MAE, training_MAE, ...]
    #    ...
    mae_tr = []

    #   List of every pair's MAE validation error at each epoch.
    #
    #   mae_va
    #     |____ [epoch 1]
    #     |        |          [pair 1]        [pair 2]
    #     |        |____ [validation_MAE, validation_MAE, ...]
    #     |____ [epoch 2]
    #     |        |          [pair 1]        [pair 2]
    #     |        |____ [validation_MAE, validation_MAE, ...]
    #    ...
    mae_va = []

    # for each set of prefixes for a training proportion
    for portion in pfs:
        # Allocate matrix to store all estimates of all 300 test user ratings
        #   from all pairs. Each element stores a 30K*3 matrix for each epoch.
        ests_total = []
        for i in range(n_epoch):
            ests_total.append(np.zeros([300*100, 3]))
        cur_row = 0

        # for prefix of each pair in the set
        for f in portion:
            tr = pd.read_csv(f + '_train.csv')  # training
            va = pd.read_csv(f + '_test.csv')   # validation
            # unshift the rating done in NMF
            tr.rating -= 10
            va.rating -= 10

            m = Mlp(tr, va, lr=0.1, batch_size=40960,
                    layer_sizes=[(16, 3), 200, 100], verbose=True)
            m.new_model()
            m.train_model(n_epoch,
                          callbacks=[SaveResults(ests_total, cur_row, m)])

            cur_row += va.shape[0]

        #final_matrix
        #            ('J1', 'J2', 'J3',...,'J100')
        # testuID1     x     x     x   ...   x
        # testuID2     x     x     x   ...   x
        #   ...        x     x     x   ...   x
        # testuID300   x     x     x   ...   x

if __name__ == '__main__':
    main()
