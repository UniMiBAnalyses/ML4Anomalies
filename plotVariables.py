from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
wSM = df.AsNumpy("w")
wpdSM = pd.DataFrame.from_dict(wSM)
#print npd

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
nEntries = 200000
npd = npd.head(nEntries)
npdBSM = npdBSM.head(nEntries)
npdBSM2 = npdBSM2.head(nEntries)
npdwBSM = npdwBSM.head(nEntries)
npdwBSM2 = npdwBSM2.head(nEntries)
wBSM = npdwBSM["w"].to_numpy()
wBSM2 = npdwBSM2["w"].to_numpy()
wpdSM = wpdSM.head(nEntries)
wSM = wpdSM["w"].to_numpy()

#t = MinMaxScaler()
#t.fit(npd)
#npd1 = t.transform(npd)
#print(npd.iloc[1])
#print(np.expand_dims(X_train[1],1))
#print npd.columns
#print npdBSM.columns
#npdBSM = t.transform(npdBSM)
#npdBSM2 = t.transform(npdBSM2)

nrows = 4
ncols = 4
fig, axes = plt.subplots(nrows=4,ncols=4)
nvar = 0
#print npd[0:,1]
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables):
            axes[i][j].hist(npd[pd_variables[nvar]],bins = 100,histtype="step",weights=wSM,color = "r",alpha=1.)
            axes[i][j].set_xlabel(pd_variables[nvar])
            axes[i][j].hist(npdBSM2[pd_variables[nvar]],bins = 100, histtype="step",weights=wBSM2,color = "b",alpha=1.)
            axes[i][j].hist(npdBSM[pd_variables[nvar]],bins = 100, histtype="step",weights=wBSM,color = "g",alpha=1.)
            nvar= nvar+1
            axes[i][j].patch.set_facecolor("w")
#ax = fig.add_subplot(111)
#ax.xaxis.grid(True, which="minor")
#ax.yaxis.grid(True, which="major")
#axes.patch.set_facecolor("w")
fig.patch.set_facecolor("w")

#npd["ptj1"].plot.hist(bins=100)
plt.show()

