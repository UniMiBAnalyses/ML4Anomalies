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

cW = 0.3 #0.3
cutLoss = 0.00004
nEntries = 200000

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll',"w"]#,'phij1', 'phij2', 'w']

dfAll = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_QUAD = ROOT.RDataFrame("SSWW_cW_QU","../ntuple_SSWW_cW_QU.root")
dfBSM_QUAD = dfBSMAll_QUAD.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_LIN = ROOT.RDataFrame("SSWW_cW_LI","../ntuple_SSWW_cW_LI.root")
dfBSM_LIN = dfBSMAll_LIN.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")

SM =pd.DataFrame.from_dict(df.AsNumpy(pd_variables))
BSM_quad=pd.DataFrame.from_dict(dfBSMAll_QUAD.AsNumpy(pd_variables))
BSM_lin=pd.DataFrame.from_dict(dfBSMAll_LIN.AsNumpy(pd_variables))
SM = SM.head(nEntries)
BSM_lin = BSM_lin.head(nEntries)
BSM_quad = BSM_quad.head(nEntries)
All_BSM = pd.concat([BSM_quad, BSM_lin], keys=['Q','L'])


#using logarithm of some variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    All_BSM[vars] = All_BSM[vars].apply(np.log10)
    SM[vars] = SM[vars].apply(np.log10)

#Rescaling weights for the Wilson coefficient

All_BSM["w"].loc["L"] = All_BSM["w"].loc["L"].to_numpy()*cW
All_BSM["w"].loc["Q"] = All_BSM["w"].loc["Q"].to_numpy()*cW*cW

weights = All_BSM["w"].to_numpy()
weights_SM = SM["w"].to_numpy()


All = pd.concat([SM,All_BSM])
weights_all = All["w"].to_numpy()
SM.drop('w',axis='columns', inplace=True)
All_BSM.drop('w',axis='columns', inplace=True)
All.drop('w',axis='columns', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(SM,SM,test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(weights_SM, weights_SM, test_size=0.2, random_state=1)
#All_BSM = All_BSM.to_numpy()
_, All_BSM_test,_,_  = train_test_split(All_BSM,All_BSM,test_size=0.2, random_state=1)
w_train, w_test,_,_ = train_test_split(weights, weights,test_size=0.2, random_state=1)
All_test = np.concatenate((All_BSM_test,X_test))
weight_test = np.concatenate((w_test, wx_test))
#weight_test = np.abs(weight_test)

t = MinMaxScaler()
t.fit(X_train)
X_test=t.transform(X_test)
All_test = t.transform(All_test)

model = tf.keras.models.load_model('vae_denselayers_withWeights_7D_latentDim_1000epoch_batchsize16_log_eventFiltered')
mylosses = LossPerBatch()
model.evaluate(X_test,X_test,batch_size=1,callbacks=[mylosses],verbose=0,sample_weight=wx_test)

mylosses_All = LossPerBatch()
model.evaluate(All_test,All_test,batch_size=1,callbacks=[mylosses_All],verbose=0,sample_weight=weight_test)

myloss = mylosses.eval_loss
myloss_All = mylosses_All.eval_loss

myloss =np.asarray(myloss)
myloss_All = np.asarray(myloss_All)
np.savetxt("lossSM_"+str(cW)+".csv", myloss,delimiter=',')
np.savetxt("lossBSM_"+str(cW)+".csv", myloss_All,delimiter=',')
#print "Eff All = ", 1.*(myloss_All>cutLoss).sum()/len(myloss_All)
#print "Eff SM = ",1.*(myloss>cutLoss).sum()/len(myloss)
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
#ax.set_ylim(ymax=10000)
ax.hist(myloss_All,bins=1000,range=(0.,0.0005),histtype="step",color="red",alpha=1.,linewidth=2,density =1,label ="BSM Loss")
ax.hist(myloss,bins=1000,range=(0.,0.0005),histtype="step",color="blue",alpha=1.,linewidth=2,density =1, label="SM Loss")
#plt.hist(myloss_BSM2,bins=100,range=(0.,0.00015),histtype="step",color="green",alpha=1.)
plt.legend()
plt.show()
ax.patch.set_facecolor("w")

t = MinMaxScaler()
t.fit(SM)
SM = t.transform(SM)
All= t.transform(All)

fig, axes = plt.subplots(nrows=4,ncols=4)
fig.patch.set_facecolor("w")
nvar = 0
nrows = 4
ncols = 4
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables):
            if pd_variables[nvar] != "w":
                axes[i][j].hist(All[0:,nvar],bins=300,range=[-0.1,1.1],density=1,weights= weights_all,histtype="step",color="red",alpha=0.5,linewidth=2,label="All")
                axes[i][j].hist(SM[0:,nvar],bins=300,range=[-0.1,1.1], density=1,weights= weights_SM,histtype="step",color="blue",alpha=0.5,linewidth=2,label="SM")
                axes[i][j].set_xlabel(pd_variables[nvar])
            nvar= nvar+1            
            #axes[i][j].xaxis.grid(True, which="major")
            #axes[i][j].yaxis.grid(True, which="major")     
#axes[3][2].legend()
plt.show()
