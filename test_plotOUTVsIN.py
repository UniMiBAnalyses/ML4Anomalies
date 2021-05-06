from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
ROOT.ROOT.EnableImplicitMT()


class LossPerBatch(tf.keras.callbacks.Callback):
    def __init__(self,**kwargs):
        self.eval_loss = []
    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        self.eval_loss = []
        #print("Start predicting; got log keys: {}".format(keys))

    def on_test_batch_end(self, batch, logs=None):
        #print("For batch {}, loss is {:7.10f}.".format(batch, logs["loss"]))
        self.eval_loss.append(logs["loss"])

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
df = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
dfBSM = ROOT.RDataFrame("SSWW_cW_QU","../ntuple_SSWW_cW_QU.root")


npy = df.AsNumpy(pd_variables)
npd =pd.DataFrame.from_dict(npy)
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    npd[vars] = npd[vars].apply(np.log10)
#print npd
wSM = df.AsNumpy("w")
wpdSM = pd.DataFrame.from_dict(wSM)

#npd = npd[(npd["ptj1"] > 200)]

npyBSM = dfBSM.AsNumpy(pd_variables)
npdBSM =pd.DataFrame.from_dict(npyBSM)
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    npdBSM[vars] = npdBSM[vars].apply(np.log10)
npywBSM = dfBSM.AsNumpy("w")
npdwBSM = pd.DataFrame.from_dict(npywBSM)

nEntries = 2000000
npd = npd.head(nEntries)
npdBSM = npdBSM.head(int(round(nEntries*0.2)))
npdwBSM = npdwBSM.head(int(round(nEntries*0.2)))
wBSM = npdwBSM["w"].to_numpy()
wpdSM = wpdSM.head(nEntries)

#print npd.columns
#print npdBSM.columns

X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()
n_inputs = npd.shape[1]
# scale data
t = MinMaxScaler()
t.fit(X_train)
#X_train = t.transform(X_train)
X_test = t.transform(X_test)
#npdBSM = t.transform(npdBSM)
model = tf.keras.models.load_model('vae_denselayers_withWeights_6_latentDim_100epoch_BatchNorm_logVars')
#model = tf.keras.models.load_model('vae_denselayers_4Dim_withWeights')
out = model.predict(X_test)
#print out[0:,0]
#print X_test[0:,0]
#diff = []
#for i in range(len(X_test[0:,0])):
#    diff.append(out[i,0]-X_test[i,0])
#setting up plots    
ax = plt.figure(figsize=(7,5), dpi=100).add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.set_ylim(ymax=2000)
fig, axes = plt.subplots(nrows=4,ncols=4)
nvar = 0
nrows = 4
ncols = 4
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables):
            axes[i][j].hist(X_test[0:,nvar],bins=500,histtype="step",color="blue",alpha=1.)
            axes[i][j].set_xlabel(pd_variables[nvar])
        
            axes[i][j].hist(out[0:,nvar],bins=500,histtype="step",color="red",alpha=1.)
            nvar= nvar+1
            axes[i][j].patch.set_facecolor("w")


plt.show()
#fig.patch.set_facecolor("w")
""" 
#
# TO ADD: plot output distributions
#
encodedBSM = encoder.predict(npdBSM)
encodedBSM2 = encoder.predict(npdBSM2)
encodedTest = encoder.predict(X_test)

fig, ((ax0, ax1,ax2)) = plt.subplots(nrows=3, ncols=1)

ax0.hist(encodedBSM[0:,0],bins=100,range=(-5.,5.),histtype="step",weights=wBSM,color="red",alpha=1.)
ax0.hist(encodedBSM[0:,1],bins=100,range=(-5.,5.),histtype="step",weights=wBSM,color="blue",alpha=1.)
ax0.hist(encodedBSM[0:,2],bins=100,range=(-5.,5.),histtype="step",weights=wBSM,color="green",alpha=1.)
ax0.hist(encodedBSM[0:,3],bins=100,range=(-5.,5.),histtype="step",weights=wBSM,color="black",alpha=1.)
ax0.set_title('BSM1')

fig.patch.set_facecolor("w")

ax1.hist(encodedBSM2[0:,0],bins=100,range=(-5.,5.),histtype="step",weights=wBSM2,color="red",alpha=1.)
ax1.hist(encodedBSM2[0:,1],bins=100,range=(-5.,5.),histtype="step",weights=wBSM2,color="blue",alpha=1.)
ax1.hist(encodedBSM2[0:,2],bins=100,range=(-5.,5.),histtype="step",weights=wBSM2,color="green",alpha=1.)
ax1.hist(encodedBSM2[0:,3],bins=100,range=(-5.,5.),histtype="step",weights=wBSM2,color="black",alpha=1.)

fig.patch.set_facecolor("w")
ax1.set_title('BSM2')

ax2.hist(encodedTest[0:,0],bins=100,range=(-5.,5.),histtype="step",weights=wxtest,color="red",alpha=1.)
ax2.hist(encodedTest[0:,1],bins=100,range=(-5.,5.),histtype="step",weights=wxtest,color="blue",alpha=1.)
ax2.hist(encodedTest[0:,2],bins=100,range=(-5.,5.),histtype="step",weights=wxtest,color="green",alpha=1.)
ax2.hist(encodedTest[0:,3],bins=100,range=(-5.,5.),histtype="step",weights=wxtest,color="black",alpha=1.)
ax2.set_title("SM")

#plt.scatter(encodedBSM2[0:,0],encodedBSM2[0:,1],c="green")
fig.patch.set_facecolor("w")

plt.show()
"""