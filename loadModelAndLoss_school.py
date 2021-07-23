from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import ROOT


#loading Loss training
lossTrain = np.loadtxt("test_vaemodel_school_loss_40Epochs.csv",delimiter=",")

epochsLossTrain = range(len(lossTrain))


ax = plt.figure(figsize=(10,5), dpi=100,facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.set_xlim(xmin =-10.,xmax=len(lossTrain)+5)
ax.plot(epochsLossTrain,lossTrain,"b",linewidth=3,alpha=0.5,label="Training Loss")
plt.ylabel("Training loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()


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


nEntries = 4000
npd = npd.head(nEntries)


X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)

# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

model = tf.keras.models.load_model('test_vaemodel_school_40Epochs')
print(model.evaluate(X_test,X_test,batch_size=32))




