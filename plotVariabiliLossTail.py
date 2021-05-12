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


pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
df = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
#df = df.Filter("mjj >0.")
npy = df.AsNumpy(pd_variables)
npd =pd.DataFrame.from_dict(npy)
dfBSM = ROOT.RDataFrame("SSWW_cW_QU","../ntuple_SSWW_cW_QU.root")
#dfBSM  = dfBSM.Filter("mjj >0.")
npyBSM = dfBSM.AsNumpy(pd_variables)
npdBSM =pd.DataFrame.from_dict(npyBSM)
nEntries = 500000
npd=npd.head(nEntries)
npdBSM = npdBSM.head(nEntries)
npywBSM = dfBSM.AsNumpy("w")
npdwBSM = pd.DataFrame.from_dict(npywBSM)
npdwBSM = npdwBSM.head(nEntries)
wBSM = npdwBSM["w"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.99, random_state=1)
n_inputs = npd.shape[1]
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
npdBSM = t.transform(npdBSM)
"""
BSM_dataset = tf.data.Dataset.from_tensor_slices((npdBSM, npdBSM))
BSM_dataset = BSM_dataset.batch(1)
"""
model = tf.keras.models.load_model('vae_denselayers_withWeights_6_latentDim_100epoch')
#How to select events with large loss

#mydict = {}
#for var in range(len(pd_variables)):
#    mydict[pd_variables[var]] = []
mse = tf.keras.losses.MeanSquaredError()

myloss_mse = []
myEvents = []
output = model.predict(npdBSM)
for i in range(nEntries):
    loss = mse(output[i],npdBSM[i]).numpy()*wBSM[i]
    myloss_mse.append(loss)
    if loss > 0.0004:
        myEvents.append(npdBSM[i])       
    


"""
myFullLoss = []
for step, (x_batch_train, y_batch_train) in enumerate(BSM_dataset):
    loss_weighted =  model.evaluate(x_batch_train,y_batch_train,verbose=0)*wBSM[step]    
    myFullLoss.append(loss_weighted)
    output = model.predict(x_batch_train)
    myloss_mse.append(mse(output,x_batch_train).numpy()*wBSM[step])    
    if mse(output,x_batch_train).numpy()*wBSM[step] > 0.004:
        myEvents.append(x_batch_train.numpy())
 
for a,b in zip(myFullLoss, myloss_mse):
    if abs(a-b)/a > 0.00001:
        print "Different losses ", a,b

"""
myEvents = np.array(myEvents)

#for i in range(len(pd_variables)):
#    print pd_variables[i], myEvents[0:,0,i] 

fig, axes = plt.subplots(nrows=4,ncols=4)
fig.patch.set_facecolor("w")
nvar = 0
nrows = 4
ncols = 4
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables):
            axes[i][j].hist(myEvents[0:,nvar],bins=300,range=[-0.1,1.1],histtype="step",color="red",alpha=0.6,linewidth=2,label="Loss > 0.0004")
            axes[i][j].hist(npdBSM[0:,nvar],bins=300,range=[-0.1,1.1],histtype="step",color="blue",alpha=0.6,linewidth=2,label="BSM")
            axes[i][j].hist(X_test[0:,nvar],bins=300,range=[-0.1,1.1],histtype="step",color="green",alpha=0.6,linewidth=2,label="SM")
            axes[i][j].set_xlabel(pd_variables[nvar])
            nvar= nvar+1            
            #axes[i][j].xaxis.grid(True, which="major")
            #axes[i][j].yaxis.grid(True, which="major")     
#axes[3][2].legend()
            

plt.show()


