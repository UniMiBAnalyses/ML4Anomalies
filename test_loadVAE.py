from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
import sys
import numpy
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
ROOT.ROOT.EnableImplicitMT()


pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
df = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
dfBSM = ROOT.RDataFrame("SSWW_cW_QU","../ntuple_SSWW_cW_QU.root")
dfBSM2 = ROOT.RDataFrame("SSWW_cHW_QU","../ntuple_SSWW_cHW_QU.root")

npy = df.AsNumpy(pd_variables)
npd =pd.DataFrame.from_dict(npy)
#print npd
wSM = df.AsNumpy("w")
wpdSM = pd.DataFrame.from_dict(wSM)

#npd = npd[(npd["ptj1"] > 200)]

npyBSM = dfBSM.AsNumpy(pd_variables)
npdBSM =pd.DataFrame.from_dict(npyBSM)
npywBSM = dfBSM.AsNumpy("w")
npdwBSM = pd.DataFrame.from_dict(npywBSM)

npyBSM2 = dfBSM2.AsNumpy(pd_variables)
npdBSM2 =pd.DataFrame.from_dict(npyBSM2)
npywBSM2 = dfBSM2.AsNumpy("w")
npdwBSM2 = pd.DataFrame.from_dict(npywBSM2)


#print npd

#Just reducing a bit the sample
#npd = npd[(npd["ptj1"] > 200)]
#npdBSM = npdBSM[(npdBSM["ptj1"] > 200)]
nEntries = 2000000000000000
npd = npd.head(nEntries*2)
npdBSM = npdBSM.head(nEntries)
npdBSM2 = npdBSM2.head(nEntries)
npdwBSM = npdwBSM.head(nEntries)
npdwBSM2 = npdwBSM2.head(nEntries)
wBSM = npdwBSM["w"].to_numpy()
wBSM2 = npdwBSM2["w"].to_numpy()
wpdSM = wpdSM.head(nEntries*2)

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
npdBSM2 = t.transform(npdBSM2)



model = tf.keras.models.load_model('vae_denselayers_4Dim_withWeights')
model.evaluate(X_test, X_test, batch_size=32,sample_weight=wxtest)
model.evaluate(npdBSM, npdBSM, batch_size=32,sample_weight=wBSM)
model.evaluate(npdBSM2, npdBSM2, batch_size=32,sample_weight=wBSM2)
encoder=tf.keras.models.load_model('encoder_test')

encodedBSM = encoder.predict(npdBSM)
#encodedBSM2 = Encoder.predict(npdBSM2)
#encodedTest = Encode.predict(X_test)

fig =plt.figure(figsize=(7,5), dpi=100)
ax = fig.add_subplot(111)
plt.hist(encodedBSM[0:,0],color="red",alpha=0.1)
plt.hist(encodedBSM[0:,1],color="blue",alpha=0.1)
plt.hist(encodedBSM[0:,2],color="green",alpha=0.1)
plt.hist(encodedBSM[0:,3],color="black",alpha=0.1)

#ax.patch.set_facecolor("w")
#fig.patch.set_facecolor("w")
#fig1 =plt.figure(figsize=(7,5), dpi=100)
#ax1 = fig.add_subplot(111)
#plt.hist(encodedTest[0:,0],color="red",alpha=0.1)
#plt.hist(encodedTest[0:,1],color="blue",alpha=0.1)
#plt.hist(encodedTest[0:,2],color="blue",alpha=0.1)
#plt.hist(encodedTest[0:,3],color="blue",alpha=0.1)


#plt.scatter(encodedBSM2[0:,0],encodedBSM2[0:,1],color="green")
#ax1.patch.set_facecolor("w")
#fig1.patch.set_facecolor("w")

plt.show()
