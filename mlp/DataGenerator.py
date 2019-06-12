import keras
import numpy as np
import pandas as pd

MAX_UID = 73421
MAX_JID = 100
UCATS = ['u' + str(i+1) for i in range(MAX_UID)]
JCATS = ['j' + str(i+1) for i in range(MAX_JID)]

np.random.seed(9999)

# Original code by Afshine Amidi & Shervine Amidi from
#   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    'Generate data batch by batch for 73,421 categorical input NN'
    def __init__(self, train, batch_size=4096, shuffle=True):
        'Initialization'
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.train.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # get the batch DF subset; index = 0, 1, 2, ...
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        df = self.train.iloc[indexes, :]

        # Dummify user and joke IDs for this DF subset
        # u_dummies = pd.get_dummies(df['uID'], prefix='u', prefix_sep='')
        # j_dummies = pd.get_dummies(df['jID'], prefix='j', prefix_sep='')
        # u_dummies = u_dummies.reindex(columns=UCATS, fill_value=0)
        # j_dummies = j_dummies.reindex(columns=JCATS, fill_value=0)
        #
        # X = pd.concat([u_dummies, j_dummies], 1).values
        # y = df.iloc[:,2].values

        X = [df.uID, df.jID]
        y = df.iloc[:, 2]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.train.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

