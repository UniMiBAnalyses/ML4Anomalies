from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt



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
nEntries = 50000

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
BSM_quad=pd.DataFrame.from_dict(dfBSM_QUAD.AsNumpy(pd_variables))
BSM_lin=pd.DataFrame.from_dict(dfBSM_LIN.AsNumpy(pd_variables))
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

#concatenating the SM + BSM part
All = pd.concat([SM,All_BSM])

#plotting correlation matrix
SM_corrM = SM.corr()
All_corrM = All.corr()
corrMatrix = SM_corrM - All_corrM
import seaborn as sn
#sn.heatmap(corrMatrix, annot=True)
sn.heatmap(All_corrM, annot=True)

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
X_train = t.transform(X_train)
X_test=t.transform(X_test)
All_test = t.transform(All_test)

model = tf.keras.models.load_model('vae_test_newModelDimenstions_MinMaxScaler')
mylosses = LossPerBatch()
mylosses_train = LossPerBatch()
model.evaluate(X_test,X_test,batch_size=1,callbacks=[mylosses],verbose=0)
model.evaluate(X_train,X_train,batch_size=1,callbacks=[mylosses_train],verbose=0)

mylosses_All = LossPerBatch()
model.evaluate(All_test,All_test,batch_size=1,callbacks=[mylosses_All],verbose=0)

myloss = mylosses.eval_loss
myloss_train = mylosses_train.eval_loss
myloss_All = mylosses_All.eval_loss

myloss =np.asarray(myloss)
myloss_All = np.asarray(myloss_All)
myloss_train =np.asarray(myloss_train)

np.savetxt("lossSM_noweigths_"+str(cW)+".csv", myloss,delimiter=',')
np.savetxt("lossBSM_noweigths_"+str(cW)+".csv", myloss_All,delimiter=',')
np.savetxt("weight_BSM_noweigths_"+str(cW)+".csv",weight_test,delimiter=',')
np.savetxt("weight_SM_noweigths_"+str(cW)+".csv",wx_test,delimiter=',')
#print "Eff All = ", 1.*(myloss_All>cutLoss).sum()/len(myloss_All)
#print "Eff SM = ",1.*(myloss>cutLoss).sum()/len(myloss)
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
#ax.set_ylim(ymax=10000)%
ax.hist(myloss_All,bins=100,range=(0.,0.1),weights=weight_test,histtype="step",color="red",alpha=.3,linewidth=2,density =1,label ="BSM Loss")
ax.hist(myloss,bins=100,range=(0.,0.1),weights=wx_test,histtype="step",color="blue",alpha=.3,linewidth=2,density =1, label="SM Test Loss")
ax.set_yscale('log')
plt.legend()

ax.patch.set_facecolor("w")

t = MinMaxScaler()
t.fit(SM)
SM = t.transform(SM)
All= t.transform(All)

bins = 100
out_SM = model.predict(X_test)
out = model.predict(All_test)
diffBSM = out-All_test
diffSM = out_SM - X_test
fig, axes = plt.subplots(nrows=4,ncols=4)
fig.patch.set_facecolor("w")
nvar = 0
nrows = 4
ncols = 4
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables)-1:                       
            axes[i][j].hist(out_SM[0:,nvar],bins=bins, density=1,weights= wx_test,range=[0.,2.],histtype="step",color="red",alpha=0.3,linewidth=1,label="SM Output")                                    
            axes[i][j].hist(X_test[0:,nvar],bins=bins, density=1,weights= wx_test,range=[0.,2.],histtype="step",color="blue",alpha=0.3,linewidth=1,label="SM Input")                                    
            axes[i][j].set_xlabel(pd_variables[nvar]) 
            axes[i][j].legend()                       
            #axes[i][j].set_xlim(xmin =-0.1,xmax=1.1)            
            #axes[i][j].set_ylim(ymin =-0.1,ymax=1.1)            
            nvar=nvar+1
            #axes[i][j].set_yscale('log')
plt.rc('legend',fontsize='xx-small')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.legend()

fig, axes = plt.subplots(nrows=4,ncols=4)
fig.patch.set_facecolor("w")
nvar = 0
nrows = 4
ncols = 4
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables)-1:                       
            axes[i][j].hist(diffBSM[0:,nvar],bins=bins, density=1,weights= weight_test,range=[-0.5,0.5],histtype="step",color="red",alpha=0.3,linewidth=1,label="BSM")                                    
            axes[i][j].hist(diffSM[0:,nvar],bins=bins, density=1,weights= wx_test,range=[-0.5,0.5],histtype="step",color="blue",alpha=0.3,linewidth=1,label="SM")                                    
            axes[i][j].set_xlabel(pd_variables[nvar]) 
            axes[i][j].legend()                       
            #axes[i][j].set_xlim(xmin =-0.1,xmax=1.1)            
            #axes[i][j].set_ylim(ymin =-0.1,ymax=1.1)            
            nvar=nvar+1
            axes[i][j].set_yscale('log')
plt.rc('legend',fontsize='xx-small')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.legend()
plt.show()