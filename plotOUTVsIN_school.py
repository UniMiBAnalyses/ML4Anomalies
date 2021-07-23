from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import ROOT


pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
dfAll = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")

npy = df.AsNumpy(pd_variables)
npd =pd.DataFrame.from_dict(npy)
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    npd[vars] = npd[vars].apply(np.log10)
#print npd
wSM = df.AsNumpy("w")
wpdSM = pd.DataFrame.from_dict(wSM)

#npd = npd[(npd["ptj1"] > 200)]

nEntries = 2000000
npd = npd.head(nEntries)
wpdSM = wpdSM.head(nEntries)

X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()
n_inputs = npd.shape[1]
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
model = tf.keras.models.load_model("test_vaemodel_school100Epochs")
out = model.predict(X_test)

fig, axes = plt.subplots(nrows=4,ncols=4)
fig.patch.set_facecolor("w")
nvar = 0
nrows = 4
ncols = 4
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables):           
            axes[i][j].hist(X_test[0:,nvar],bins=500, density =1, weights = wxtest,range=[-0.1,1.2],histtype="step",color="blue",alpha=0.6,linewidth=2,label="Input")
            axes[i][j].set_xlabel(pd_variables[nvar])            
            axes[i][j].hist(out[0:,nvar],density =1, bins=500,range=[-0.1,1.2],weights = wxtest,histtype="step",color="red",alpha=0.6,linewidth=2,label="500 epochs 6D")
            nvar=nvar+1
plt.show()