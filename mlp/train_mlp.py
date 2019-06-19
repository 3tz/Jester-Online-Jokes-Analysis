from mlp import *

tr = pd.read_csv('./cvout/90_1_train.csv')
va = pd.read_csv('./cvout/90_1_test.csv')

tr.rating -= 10
va.rating -= 10

m = Mlp(tr, va, lr=0.1, batch_size=4096, layer_sizes=[(16, 3), 200, 100],
        verbose=True)
m.new_model()
m.train_model()