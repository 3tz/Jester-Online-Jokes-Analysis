import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

nmf_mae_tr = pd.read_csv('csv/nmf_mae_tr.csv')
nmf_mae_va = pd.read_csv('csv/nmf_mae_va.csv')
nmf_mae_tr_extra = pd.read_csv('csv/nmf_mae_tr_extra_90.csv')
nmf_mae_va_extra = pd.read_csv('csv/nmf_mae_va_extra_90.csv')

extra90_tr = np.hstack([nmf_mae_tr.iloc[2,:].values, nmf_mae_tr_extra.values[0]])
extra90_va = np.hstack([nmf_mae_va.iloc[2,:].values, nmf_mae_va_extra.values[0]])


plt.figure()
plt.xlabel('Rank',  fontsize = 15)
plt.ylabel('MAE', fontsize = 15)
x = nmf_mae_tr.columns.astype('int')

x_extra90 = np.hstack([nmf_mae_tr.columns.values,
                       nmf_mae_tr_extra.columns.values]).astype('int')


plt.plot(x, nmf_mae_tr.iloc[0,:], 'g--', label='Training MAE; 30%')
plt.plot(x, nmf_mae_va.iloc[0,:], 'g-', label='Validation MAE; 30%')
plt.plot(x, nmf_mae_tr.iloc[1,:], '--', color='#FFA500', label='Training MAE; 60%')
plt.plot(x, nmf_mae_va.iloc[1,:], '-', color='#FFA500', label='Validation MAE; 60%')
plt.plot(x_extra90, extra90_tr, 'r--', label='Training MAE; 90%')
plt.plot(x_extra90, extra90_va, 'r-', label='Validation MAE; 90%')

#plt.xlim(math.floor(min(df_test[f].values)*0.8), math.ceil(max(df_test[f].values)*1.2))
#plt.ylim(math.floor(min(df_test['mpg'].values)*0.8), math.ceil(max(df_test['mpg'].values)*1.2))
plt.legend(bbox_to_anchor=(1, 0.7), loc='upper right')
plt.title('NMF: MAE vs Rank')
plt.savefig('fig/nmf_mae.png')
plt.clf()



opm30 = x[nmf_mae_va.iloc[0,:].values.argmin()]
opm60 = x[nmf_mae_va.iloc[1,:].values.argmin()]
opm90 = x_extra90[extra90_va.argmin()]


def nABC(tcv):
    cols = ['uID'] + ['rec' + str(i+1) for i in range(100)]
    print('Ternary Count: a: {0:0.3f}%  b: {1:0.3f}%  c: {2:0.3f}%'
             .format(np.sum(np.sum(tcv[cols[1:]] == 'a'))/30000 * 100,
                     np.sum(np.sum(tcv[cols[1:]] == 'b'))/30000 * 100,
                     np.sum(np.sum(tcv[cols[1:]] == 'c'))/30000 * 100))

dn = '../nmf/output/'

tcv30 = pd.read_csv(dn + '30_10/tcv_nmf.csv')
tcv60 = pd.read_csv(dn + '60_20/tcv_nmf.csv')
tcv90 = pd.read_csv(dn + '90_30/tcv_nmf.csv')

nABC(tcv30)
nABC(tcv60)
nABC(tcv90)



######### MLP #########

N = 30000

n_ratings = {'30': np.matrix([21000/N, 9000/N]),
             '60': np.matrix([9900/N, 10200/N, 9900/N]),
             '90': np.matrix([2400/N, 2700/N, 2700/N, 2700/N, 2700/N, 2400/N,
                              2400/N, 2400/N, 2400/N, 2400/N, 2400/N, 2400/N])}

# Matrix of every pair's validation MAE at each epoch.
#
#         epoch1 epoch2 ...
# pair1     x      x   ...
# pair2     x      x   ...
#  ...
dn = '../mlp/pkl/'

with open(dn + 'mae_tr_30_100_current.pkl', 'rb') as f:
    mae_tr30 = pickle.load(f)
with open(dn + 'mae_tr_60_100_current.pkl', 'rb') as f:
    mae_tr60 = pickle.load(f)
with open(dn + 'mae_tr_90_150_current.pkl', 'rb') as f:
    mae_tr90 = pickle.load(f)

with open(dn + 'mae_va_30_100_current.pkl', 'rb') as f:
    mae_va30 = pickle.load(f)
with open(dn + 'mae_va_60_100_current.pkl', 'rb') as f:
    mae_va60 = pickle.load(f)
with open(dn + 'mae_va_90_150_current.pkl', 'rb') as f:
    mae_va90 = pickle.load(f)

mul_tr30 = np.mean(mae_tr30, 0)
mul_tr60 = np.mean(mae_tr60, 0)
mul_tr90 = np.mean(mae_tr90, 0)

mul_va30 = np.matmul(n_ratings['30'], mae_va30).A1
mul_va60 = np.matmul(n_ratings['60'], mae_va60).A1
mul_va90 = np.matmul(n_ratings['90'], mae_va90).A1

x = np.arange(100)
x_90 = np.arange(150)

plt.figure()
plt.xlabel('Epoch',  fontsize = 15)
plt.ylabel('MAE', fontsize = 15)
plt.plot(x, mul_tr30, 'g--', label='Training MAE; 30%')
plt.plot(x, mul_va30, 'g-', label='Validation MAE; 30%')
plt.plot(x, mul_tr60, '--', color='#FFA500', label='Training MAE; 60%')
plt.plot(x, mul_va60, '-', color='#FFA500', label='Validation MAE; 60%')
plt.plot(x_90, mul_tr90, 'r--', label='Training MAE; 90%')
plt.plot(x_90, mul_va90, 'r-', label='Validation MAE; 90%')

#plt.xlim(math.floor(min(df_test[f].values)*0.8), math.ceil(max(df_test[f].values)*1.2))
#plt.ylim(math.floor(min(df_test['mpg'].values)*0.8), math.ceil(max(df_test['mpg'].values)*1.2))
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 8})
plt.title('MLP: MAE vs Epoch')
plt.savefig('fig/mlp_mae.png')

x_ub = 60
plt.xlim(-2, x_ub)
plt.legend(loc='lower left', prop={'size': 8})
plt.savefig('fig/mlp_mae_xlim{}.png'.format(x_ub))
#plt.show()
plt.clf()

# for i in range(150):
#     if i < 100:
#         print('{0} & {1:0.3f} & {2:0.3f} & {3:0.3f} \\\\'.format(i+1, mul_tr30[i], mul_tr60[i], mul_tr90[i]))
#     else:
#         print('{0} & & & {1:0.3f} \\\\'.format(i+1, mul_tr90[i]))
#
#
# for i in range(150):
#     if i < 100:
#         print('{0} & {1:0.3f} & {2:0.3f} & {3:0.3f} \\\\'.format(i+1, mul_va30[i], mul_va60[i], mul_va90[i]))
#     else:
#         print('{0} & & & {1:0.3f} \\\\'.format(i+1, mul_va90[i]))
#

dn = '../mlp/output/'

tcv30_mlp = pd.read_csv(dn + '30_5/tcv_mlp.csv')
tcv60_mlp = pd.read_csv(dn + '60_6/tcv_mlp.csv')
tcv90_mlp = pd.read_csv(dn + '90_7/tcv_mlp.csv')


nABC(tcv90_mlp)
nABC(tcv60_mlp)
nABC(tcv30_mlp)



