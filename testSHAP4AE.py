from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
#import sys
import numpy as np
import pandas as pd
import tensorflow as tf
#from matplotlib import pyplot as plt

#from VAE_model import *

ROOT.ROOT.EnableImplicitMT()

def sort_by_absolute(df, index):
    df_abs = df.apply(lambda x: abs(x))
    df_abs = df_abs.sort_values('reconstruction_loss', ascending = False)
    df = df.loc[df_abs.index,:]
    return df


cW = 0.5 #0.3
nEntries = 2000

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll',"w"]#,'phij1', 'phij2']

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

#concatenating the SM + BSM part
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

t = MinMaxScaler()
t.fit(X_train)
#X_test=t.transform(X_test)
All_test = t.transform(All_test)
All_BSM_test = t.transform(All_BSM_test)

model = tf.keras.models.load_model('vae_denselayers_withWeights_6D_latentDim_1000epoch_batchsize16_log_eventFiltered')
output = model.predict(All_BSM_test)
rec_err = np.linalg.norm(All_BSM_test - output, axis = 1)
idx = list(rec_err).index(max(rec_err))
#print idx
df = pd.DataFrame(data = All_BSM_test[idx], index = All.columns, columns = ['reconstruction_loss'])
#print(sort_by_absolute(df, idx).T)
top_5_features = sort_by_absolute(df, idx).iloc[:5,:]
print(top_5_features.T)