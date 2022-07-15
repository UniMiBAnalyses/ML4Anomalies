from sklearn import datasets
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


#taking the model
#from VAE_model_extended_moreDKL import *
from VAE_testReturnLoss import *
from matplotlib import pyplot as plt

import ROOT

#
# variable from the nutple
#
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
kinematicFilter = "ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200"
dfSM = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
dfSM = dfSM.Filter(kinematicFilter)
dfBSM = ROOT.RDataFrame("SSWW_cW_QU","../ntuple_SSWW_cW_QU.root")
dfBSM = dfBSM.Filter(kinematicFilter)
dfBSM2 = ROOT.RDataFrame("SSWW_cqq3_QU","../ntuple_SSWW_cqq3_QU.root")
dfBSM2 = dfBSM2.Filter(kinematicFilter)

np_SM = dfSM.AsNumpy(pd_variables)
wSM = dfSM.AsNumpy("w")
npd =pd.DataFrame.from_dict(np_SM)
wpdSM = pd.DataFrame.from_dict(wSM)

np_BSM = dfBSM.AsNumpy(pd_variables)
wBSM = dfBSM.AsNumpy("w")
npd_BSM =pd.DataFrame.from_dict(np_BSM)
wpdBSM = pd.DataFrame.from_dict(wBSM)

np_BSM2 = dfBSM2.AsNumpy(pd_variables)
wBSM2 = dfBSM2.AsNumpy("w")
npd_BSM2 =pd.DataFrame.from_dict(np_BSM2)
wpdBSM2 = pd.DataFrame.from_dict(wBSM2)

nEntries = 300000
npd = npd.head(nEntries)
npd_BSM = npd_BSM.head(nEntries)
npd_BSM2 = npd_BSM2.head(nEntries)
wpdSM = wpdSM.head(nEntries)
wpdBSM = wpdBSM.head(nEntries)
wpdBSM2 = wpdBSM2.head(nEntries)
#to be done for all the pt and mass and met variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    npd[vars] = npd[vars].apply(numpy.log10)
    npd_BSM[vars] = npd_BSM[vars].apply(numpy.log10)
    npd_BSM2[vars] = npd_BSM2[vars].apply(numpy.log10)

Y_true = np.full(nEntries,0)
Y_true_BSM = np.full(nEntries,1)
Y_true_BSM2 = np.full(nEntries,1)

X_train, X_test, y_train, y_test = train_test_split(npd, Y_true, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)

BSM_train, BSM_test, y_BSM_train, y_BSM_test = train_test_split(npd_BSM, Y_true_BSM, test_size=0.2, random_state=1)
wBSM_train, wBSM_test, _ , _ = train_test_split(wpdBSM, wpdBSM, test_size=0.2, random_state=1)
BSM2_train, BSM2_test, y_BSM2_train, y_BSM2_test = train_test_split(npd_BSM2, Y_true_BSM2, test_size=0.2, random_state=1)
wBSM2_train, wBSM2_test, _ , _ = train_test_split(wpdBSM2, wpdBSM2, test_size=0.2, random_state=1)

#print wx_train,X_train
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()
wBSM = wBSM_train["w"].to_numpy()
wBSMtest = wBSM_test["w"].to_numpy()

wBSM2 = wBSM2_train["w"].to_numpy()
wBSM2test = wBSM2_test["w"].to_numpy()
# scale data
t = MinMaxScaler()
#t = StandardScaler()
t.fit(X_train)
X_test = t.transform(X_test)
BSM_test = t.transform(BSM_test)
BSM2_test = t.transform(BSM2_test)

n_inputs = npd.shape[1]
original_dim = n_inputs

intermediate_dim = 20 #50 by default
input_dim = 10 #was 20 in default
half_input = 7 #was 20 in the newTest
latent_dim = 5 #was 3 for optimal performance
epochs = 200 #100
batchsize=64 #32
nameExtenstion = str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs)+"_"+str(batchsize)

latent_dim2 = 7
nameExtenstion2 = str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim2)+"_"+str(epochs)+"_"+str(batchsize)

vae = tf.keras.models.load_model('vae_newModelUsingKL_Reco_Loss_newWayToAddUpSamples_classWights_20_10_7_7_100_64')
vae2 = tf.keras.models.load_model('vae_newModelUsingKL_totLoss_100Nodes_20_10_7_7_100_16')
#concatenating SM and BSM
label_test = np.concatenate((y_test,y_BSM_test))
weights_test=np.concatenate((wxtest,wBSMtest))

label_test2 = np.concatenate((y_test,y_BSM2_test))
weights_test2=np.concatenate((wxtest,wBSM2test))

output_SM = vae.predict(X_test)
output_BSM = vae.predict(BSM_test)
output_BSM2 = vae.predict(BSM2_test)

output =  np.concatenate((output_SM,output_BSM))
output2 =  np.concatenate((output_SM,output_BSM2))

output_vae2_SM = vae2.predict(X_test)
output_vae2_BSM = vae2.predict(BSM_test)
output_vae2_BSM2 = vae2.predict(BSM2_test)

output_vae2 =  np.concatenate((output_vae2_SM,output_vae2_BSM))
output2_vae2 =  np.concatenate((output_vae2_SM,output_vae2_BSM2))

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(label_test,output)
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(label_test2,output2)
print('roc_auc_score for VAE_cW_QU: ', roc_auc_score(label_test,output))
print('roc_auc_score for VAE_cqq3_QU: ', roc_auc_score(label_test2,output2))

false_positive_rate1_vae2, true_positive_rate1_vae2, threshold1_vae2 = roc_curve(label_test,output_vae2)
false_positive_rate2_vae2, true_positive_rate2_vae2, threshold2_vae2 = roc_curve(label_test2,output2_vae2)
print('roc_auc_score for VAE2_cW_QU: ', roc_auc_score(label_test,output_vae2))
print('roc_auc_score for VAE2_cqq3_QU: ', roc_auc_score(label_test2,output2_vae2))


bins=100
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(output_SM,bins=bins, weights=wxtest,density=1,range=[0.,1.],histtype="step",color="red",alpha=0.6,linewidth=2,label=nameExtenstion+"_SM Output")                        
ax.hist(output_BSM,bins=bins, weights=wBSMtest, density=1,range=[0.,1.],histtype="step",color="blue",alpha=0.6,linewidth=2,label=nameExtenstion+"_cW_QU Output")                        
ax.hist(output_BSM2,bins=bins, weights=wBSM2test,density=1,range=[0.,1.],histtype="step",color="green",alpha=0.6,linewidth=2,label=nameExtenstion+"_cqq3 Output")                        

ax.hist(output_vae2_SM,bins=bins, weights=wxtest,density=1,range=[0.,1.],histtype="step",color="orange",alpha=0.6,linewidth=2,label="VAE2_SM Output")                        
ax.hist(output_vae2_BSM,bins=bins, weights=wBSMtest, density=1,range=[0.,1.],histtype="step",color="black",alpha=0.6,linewidth=2,label="VAE2_cW_QU Output")                        
ax.hist(output_vae2_BSM2,bins=bins, weights=wBSM2test,density=1,range=[0.,1.],histtype="step",color="purple",alpha=0.6,linewidth=2,label="VAE2_cqq3 Output")                        
plt.legend()  

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - VAE')
plt.plot(false_positive_rate1, true_positive_rate1,label=nameExtenstion+"_cW_QU")
plt.plot(false_positive_rate2, true_positive_rate2,label=nameExtenstion+"_cqq3_QU")
plt.plot(false_positive_rate1_vae2, true_positive_rate1_vae2,label="VAE2_cW_QU")
plt.plot(false_positive_rate2_vae2, true_positive_rate2_vae2,label="VAE2_cqq3_QU")
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
