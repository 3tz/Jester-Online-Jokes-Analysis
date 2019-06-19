import os
from datetime import datetime
import numpy as np
from tensorflow import set_random_seed
from tensorflow import logging

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, LeakyReLU, ELU, Activation
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K

from DataGenerator import *

np.random.seed(9999)
set_random_seed(9999)

logging.set_verbosity(logging.ERROR)        # suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # suppress TF warnings

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # Set path
np.set_printoptions(linewidth=250, threshold=np.nan, suppress=True)

# Reference:
#   - https://github.com/hexiangnan/neural_collaborative_filtering
#   - https://github.com/alexvlis/movie-recommendation-system

class Mlp(object):
    def __init__(self, train, val, lr, batch_size=4096, activation=LeakyReLU(),
                 layer_sizes=[(16, 3), 200, 100], dropout=False, bNorm=True,
                 regularizer=None, verbose=False, useGen=False, shuffle=False,
                 callbacks=None):
        """
        Initialize parameters required for training and testing the network.
        Structure:

          [User ID; int]           [Joke ID; int]
                 |                        |
                 V                        V
        [User ID Embedding]       [Joke ID Embedding]
      length: @layer_sizes[0]   length: @layer_sizes[1]
                 |                        |
                   \                    /
                     \                /
                    [Concatenated Layer]
           length: @layer_sizes[0] + @layer_sizes[1]
                            |
                            V
                     [Hidden Layer 1]
                            |
                            V
                           ...
                     [Hidden Layer n]
                            |
                            V
                         [Output]

        Arguments:
            - train: pandas.dataframe
                Dataframe that contains the training set in the n x 3 format.
            - val: pandas.dataframe
                Dataframe that contains the validation set in the n x 3 format.
            - lr: int
                Learning rate of the network.
            - batch_size: int, default 4096
                Size of each mini-batch.
            - activation: func, default LeakyReLU()
                Activation function of all hidden layers.
            - layer_sizes: list, default [(16, 3), 200, 100]
                List of the sizes of each layer.
                Note: the 1st element is a tuple that contains the sizes of the
                 embedding layers for user IDs and joke IDs correspondingly.
            - dropout: boolean, default False
                Incidates whether to use dropout for hidden layers.
            - bNorm: boolean, default True
                Indicates whether to use Batch Normalization on all layers.
            - regularizer: func, default None
                Regularizer to use.
            - verbose: boolean, default False
                Indicates whether to print out attributes before training.
            - useGen: boolean, default False
                fit_generator() is used if True; otherwise, fit() is used.
            - shuffle: boolean, default False
                Whether to shuffle the order of dataset.
            - callbacks: list, default None
                List of keras.callbacks.Callback objects to run
        Returns:
            - None
        """
        self.train = train
        self.val = val
        self.lr = lr
        self.bs = batch_size
        self.a = activation
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.bNorm = bNorm
        self.reg = regularizer
        self.verbose = verbose
        self.useGen = useGen
        self.shuffle = shuffle
        self.callbacks = callbacks

        self.model = None
        self.trGen = DataGenerator(train, batch_size, shuffle)

    def new_model(self):
        """
        Construct model structure based on attributes.
        """

        emb_dim = self.layer_sizes[0] # embedding dimension

        input_uid = Input(shape=(1,), dtype='int32', name='input_uid')
        input_jid = Input(shape=(1,), dtype='int32', name='input_jid')

        embedding_uid = Embedding(input_dim=self.train['uID'].max() + 1,
                                  output_dim=emb_dim[0], input_length=1,
                                  embeddings_regularizer=self.reg,
                                  name='embedding_uid')
        embedding_jid = Embedding(input_dim=self.train['jID'].max() + 1,
                                  output_dim=emb_dim[1], input_length=1,
                                  embeddings_regularizer=self.reg,
                                  name='embedding_jid')


        latent_uid = embedding_uid(input_uid)
        latent_jid = embedding_jid(input_jid)

        if self.bNorm:
            latent_uid = BatchNormalization()(latent_uid)
            latent_jid = BatchNormalization()(latent_jid)

        latent_uid = Flatten()(latent_uid)
        latent_jid = Flatten()(latent_jid)

        # The actual "input layer" of our MLP with a size of layer_sizes[0]
        vector = concatenate([latent_uid, latent_jid])

        layer = vector
        for i in range(1, len(self.layer_sizes)):
            n = self.layer_sizes[i]
            # connect layers
            z = Dense(n, kernel_regularizer=self.reg,
                      name='hidden_%d' % i)(layer)
            z = self.a(z)
            if self.bNorm:
                z = BatchNormalization()(z)
            if self.dropout:
                z = Dropout(0.5)(z)
            layer = z

        out = Dense(1, activation='linear', name='output')(layer)

        if self.bNorm:
            out = BatchNormalization()(out)

        self.model = Model(inputs=[input_uid, input_jid], outputs=out)


    def train_model(self, n_epoch=150):
        """
        Train self.model with dataset stored in attributes.
        """

        if self.verbose:
            print("Learning Rate:\t", self.lr)
            print("Batch Size:\t", self.bs)
            print("Activation:\t", self.a)
            print("Layer Sizes:\t", self.layer_sizes)
            print("Dropout:\t", self.dropout)
            print("Batch Norm:\t", self.bNorm)
            print("Regularizer:\t", self.reg)
            print("Callbacks:\t", self.callbacks)

        # Time-stamp for saving
        ts = hex(int((datetime.now()).timestamp()))[2:]

        self.model.compile(optimizer=Adam(lr=self.lr, decay=.001),
                           loss='mean_squared_error', metrics=['mae'])

        valX = [self.val.uID, self.val.jID]
        valy = self.val.iloc[:, 2]

        if self.useGen:
            self.hist = self.model.fit_generator(generator=self.trGen,
                                                 validation_data=(valX, valy),
                                                 epochs=n_epoch, verbose=1,
                                                 use_multiprocessing=True,
                                                 workers=5,
                                                 callbacks=self.callbacks)
        else:
            self.hist = self.model.fit([self.train.uID, self.train.jID],
                                       self.train.iloc[:, 2], epochs=n_epoch,
                                       verbose=1, validation_data=(valX, valy),
                                       batch_size=self.bs,
                                       callbacks=self.callbacks)
