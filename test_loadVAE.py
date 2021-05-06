from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

#from VAE_model import *

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
#for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
#       'ptl2', 'ptll']:
    #npd[vars] = npd[vars].apply(np.log10)
#print npd
wSM = df.AsNumpy("w")
wpdSM = pd.DataFrame.from_dict(wSM)

#npd = npd[(npd["ptj1"] > 200)]

npyBSM = dfBSM.AsNumpy(pd_variables)
npdBSM =pd.DataFrame.from_dict(npyBSM)
#for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
#       'ptl2', 'ptll']:
    #npdBSM[vars] = npdBSM[vars].apply(np.log10)
#print npd
npywBSM = dfBSM.AsNumpy("w")
npdwBSM = pd.DataFrame.from_dict(npywBSM)


#print npd

#Just reducing a bit the sample
#npd = npd[(npd["ptj1"] > 200)]
#npdBSM = npdBSM[(npdBSM["ptj1"] > 200)]
nEntries = 400000
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
X_train = t.transform(X_train)
X_test = t.transform(X_test)
npdBSM = t.transform(npdBSM)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test))
test_dataset = test_dataset.batch(1)
mylosses = LossPerBatch()
model = tf.keras.models.load_model('vae_denselayers_withWeights_6_latentDim_100epoch_BatchNorm')
#model.evaluate(X_test,X_test,batch_size=1,callbacks=[mylosses],verbose=0,sample_weight=wxtest)
for step, (x_batch_train, y_batch_train) in enumerate(test_dataset):
    loss_weighted =  model.evaluate(x_batch_train,y_batch_train,verbose=0)*wxtest[step]
    if loss_weighted > 0.00001:
        print x_batch_train

mylosses_BSM = LossPerBatch()
model.evaluate(npdBSM, npdBSM, batch_size=1,sample_weight=wBSM,callbacks=[mylosses_BSM],verbose=0)

myloss = mylosses.eval_loss
myloss_BSM = mylosses_BSM.eval_loss
np.savetxt("lossSM.csv", myloss,delimiter=',')
np.savetxt("lossBSM.csv", myloss_BSM,delimiter=',')

#print myloss_BSM
#myloss_BSM2 = mylosses_BSM2.eval_loss
ax = plt.figure(figsize=(7,5), dpi=100).add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.set_ylim(ymax=2000)
ax.hist(myloss,bins=1000,range=(0.,0.0005),histtype="step",color="blue",alpha=1.)
ax.hist(myloss_BSM,bins=1000,range=(0.,0.0005),histtype="step",color="red",alpha=1.)
#plt.hist(myloss_BSM2,bins=100,range=(0.,0.00015),histtype="step",color="green",alpha=1.)
plt.show()
ax.patch.set_facecolor("w")
#fig.patch.set_facecolor("w")

#
# TO ADD: plot output distributions
#
encoder = tf.keras.models.load_model('encoder_6_latentDim_BatchNorm_logVars')
encodedBSM = encoder.predict(npdBSM)
encodedTest = encoder.predict(X_test)

fig, ((ax0, ax1,ax2)) = plt.subplots(nrows=3, ncols=1)

ax0.hist(encodedBSM[0:,0],bins=100,range=(-5.,5.),histtype="step",weights=wBSM,color="red",alpha=1.)
ax0.hist(encodedBSM[0:,1],bins=100,range=(-5.,5.),histtype="step",weights=wBSM,color="blue",alpha=1.)
ax0.hist(encodedBSM[0:,2],bins=100,range=(-5.,5.),histtype="step",weights=wBSM,color="green",alpha=1.)
ax0.hist(encodedBSM[0:,3],bins=100,range=(-5.,5.),histtype="step",weights=wBSM,color="black",alpha=1.)
ax0.set_title('BSM1')
ax0.patch.set_facecolor("w")
fig.patch.set_facecolor("w")

ax2.hist(encodedTest[0:,0],bins=100,range=(-5.,5.),histtype="step",weights=wxtest,color="red",alpha=1.)
ax2.hist(encodedTest[0:,1],bins=100,range=(-5.,5.),histtype="step",weights=wxtest,color="blue",alpha=1.)
ax2.hist(encodedTest[0:,2],bins=100,range=(-5.,5.),histtype="step",weights=wxtest,color="green",alpha=1.)
ax2.hist(encodedTest[0:,3],bins=100,range=(-5.,5.),histtype="step",weights=wxtest,color="black",alpha=1.)
ax2.set_title("SM")

#plt.scatter(encodedBSM2[0:,0],encodedBSM2[0:,1],c="green")
ax2.patch.set_facecolor("w")
fig.patch.set_facecolor("w")

plt.show()
