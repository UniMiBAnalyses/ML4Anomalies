from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib import pyplot as plt

import ROOT
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
df = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
#dfBSM = ROOT.RDataFrame("SSWW_cW_QU","../ntuple_SSWW_cW_QU.root")
#dfBSM2 = ROOT.RDataFrame("SSWW_cW_LI","../ntuple_SSWW_cW_LI.root")

#Transforming it into a DataFrame choosing which variables to use
npy = df.AsNumpy(pd_variables)
npd =pd.DataFrame.from_dict(npy)

wSM = df.AsNumpy("w")
wpdSM = pd.DataFrame.from_dict(wSM)

nEntries = 2000000
npd = npd.head(nEntries)
wSM = wpdSM["w"].to_numpy()

#how it looks like:
print(npd)

#Easy to plot
_ = npd.hist(figsize=(20, 14),histtype="step")

#normalizing the variables
t = MinMaxScaler()
t.fit(npd)
npd = t.transform(npd)

# a better looking plot

nrows = 4
ncols = 4
fig, axes = plt.subplots(nrows=4,ncols=4)
nvar = 0
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables):
            axes[i][j].hist(npd[0:,nvar],bins = 50,range = (0.,1.),histtype="step",weights=wSM,color = "r",alpha=1.,density=1)
            axes[i][j].set_xlabel(pd_variables[nvar])
            nvar= nvar+1
            axes[i][j].patch.set_facecolor("w")
#ax.yaxis.grid(True, which="major")
fig.subplots_adjust(hspace=0.7, right=0.9)
plt.show()

