import keras
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model
from keras.models import load_model
from keras import optimizers
import numpy as np
import pandas as pd

tr = pd.read_csv('./cvout/90_12_train.csv')
va = pd.read_csv('./cvout/90_12_test.csv')

n_users = len(tr.uID.unique())
n_jokes = len(tr.jID.unique())

model = load_model("model_90_12_epoch50.h5")


predictions = model.predict([va.uID, va.jID])
predictions = np.array([a[0] for a in predictions])
mean_abs_error = np.mean(np.abs(predictions.T - va.rating))
print "MAE for 90_12_test epoch 50 = ", mean_abs_error
