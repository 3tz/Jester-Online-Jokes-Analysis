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

joke_input = Input(shape=[1], name="Joke-Input")
joke_embedding = Embedding(n_jokes + 1, 4, name="Joke-Embedding")(joke_input)
joke_vec = Flatten(name="Flatten-Jokes")(joke_embedding)
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users + 1, 8, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)
conc = keras.layers.concatenate([joke_vec, user_vec])
#drop = keras.layers.Dropout(0.5,name='Dropout')(conc)
fc1 = Dense(500, activation='relu')(conc)
#drop1 = keras.layers.Dropout(0.7,name='Dropout1')(fc1)
fc2 = Dense(250, activation='relu')(fc1)
#drop2 = keras.layers.Dropout(0.2,name='Dropout2')(fc2)
fc3 = Dense(10, activation='relu')(fc2)

out = Dense(1, activation = 'relu')(fc3)
#model = Model([user_input, joke_input], prod)
model = Model([user_input, joke_input], out)
adam = optimizers.Adam(lr=0.025, decay=0.01)
model.compile(loss='mean_absolute_error', optimizer=adam)


history = model.fit([tr.uID, tr.jID], tr.rating, epochs=50, verbose=1, batch_size=800)
#to save the model
model.save('model_90_12_epoch50.h5')
#to load a model
#model = load_model("model_90_12_epoch50.h5")

predictions = model.predict([va.uID, va.jID])
predictions = np.array([a[0] for a in predictions])
mean_abs_error = np.mean(np.abs(predictions.T - va.rating))
print "MAE for 90_12_test epoch 50 = ", mean_abs_error
