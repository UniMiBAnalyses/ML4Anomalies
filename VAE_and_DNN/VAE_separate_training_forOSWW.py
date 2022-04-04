from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import sys
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

#taking the model
#from VAE_model_extended_moreDKL import *
from VAE_separateT import *
#from testDNN import *
from matplotlib import pyplot as plt

import ROOT
#ROOT.ROOT.EnableImplicitMT()

path_to_ntuple = "/gwpool/users/glavizzari/Downloads/ntuplesOSWW"
#
# variable from the nutple
#
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
kinematicFilter = "ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200"
dfSM = ROOT.RDataFrame("OSWW_SM",path_to_ntuple+"/ntuple_OSWW_SM.root")
dfSM = dfSM.Filter(kinematicFilter)
dfSMQCD = ROOT.RDataFrame("OSWWQCD_SM",path_to_ntuple+"/ntuple_OSWWQCD_SM.root")
dfSMQCD = dfSMQCD.Filter(kinematicFilter)
dfBSM = ROOT.RDataFrame("OSWW_cW_QU",path_to_ntuple+"/ntuple_OSWW_cW_QU.root")
dfBSM_LI = ROOT.RDataFrame("OSWW_cW_LI",path_to_ntuple+"/ntuple_OSWW_cW_LI.root")
dfBSM = dfBSM.Filter(kinematicFilter)
dfBSM_LI = dfBSM_LI.Filter(kinematicFilter)

np_SM = dfSM.AsNumpy(pd_variables)
wSM = dfSM.AsNumpy("w")
np_SMQCD = dfSMQCD.AsNumpy(pd_variables)
wSMQCD = dfSMQCD.AsNumpy("w")
npd =pd.DataFrame.from_dict(np_SM)
wpdSM = pd.DataFrame.from_dict(wSM)
npdQCD =pd.DataFrame.from_dict(np_SMQCD)
wpdSMQCD = pd.DataFrame.from_dict(wSMQCD)

np_BSM = dfBSM.AsNumpy(pd_variables)
wBSM = dfBSM.AsNumpy("w")
npd_BSM =pd.DataFrame.from_dict(np_BSM)
wpdBSM = pd.DataFrame.from_dict(wBSM)

np_BSM_LI = dfBSM_LI.AsNumpy(pd_variables)
wBSM_LI = dfBSM_LI.AsNumpy("w")
npd_BSM_LI =pd.DataFrame.from_dict(np_BSM_LI)
wpdBSM_LI = pd.DataFrame.from_dict(wBSM_LI)

nEntries = 60000
npd = npd.head(nEntries)
npdQCD = npdQCD.head(nEntries)
npd_BSM = npd_BSM.head(nEntries)
npd_BSM_LI = npd_BSM_LI.head(nEntries)
wpdSM = wpdSM.head(nEntries)
wpdSMQCD = wpdSMQCD.head(nEntries)
wpdBSM_LI = wpdBSM_LI.head(nEntries)

#to be done for all the pt and mass and met variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    npd[vars] = npd[vars].apply(numpy.log10)
    npdQCD[vars] = npdQCD[vars].apply(numpy.log10)
    npd_BSM[vars] = npd_BSM[vars].apply(numpy.log10)
    npd_BSM_LI[vars] = npd_BSM_LI[vars].apply(numpy.log10)

Y_true = np.full(2*nEntries,0)
Y_true_BSM = np.full(2*nEntries,1)
#Y_true_BSM = np.full(nEntries,1)

#concatenating SM and BSM
labels = np.concatenate((Y_true,Y_true_BSM))
samples = np.concatenate((npd,npdQCD,npd_BSM,npd_BSM_LI))
weights = np.concatenate((wpdSM,wpdSMQCD,wpdBSM,wpdBSM_LI))
#samples = np.concatenate((npd,npd_BSM))
#weights = np.concatenate((wpdSM,wpdBSM))
totSM = np.concatenate((npd,npdQCD))
X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=1) #random_state = 1
weight_train,weight_test,_,_=train_test_split(weights, weights, test_size=0.2, random_state=1) #random_state = 1
SM_train,SM_test,true_train,true_test = train_test_split(totSM, Y_true, test_size=0.2, random_state=1) #random_state = 1
BSM_train,BSM_test,_,_ = train_test_split(npd_BSM, npd_BSM, test_size=0.2, random_state=2) #random_state = 1
#wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
SM_only_train = np.concatenate((SM_train,SM_train))
SM_only_test = np.concatenate((SM_test,SM_test))
#BSM_train, BSM_test, y_BSM_train, y_BSM_test = train_test_split(npd_BSM, Y_true_BSM, test_size=0.2, random_state=1)
#wBSM_train, wBSM_test, _ , _ = train_test_split(wpdBSM, wpdBSM, test_size=0.2, random_state=1)
#print wx_train,X_train
#wx = wx_train["w"].to_numpy()
#wxtest = wx_test["w"].to_numpy()
#wBSM = wBSM_train["w"].to_numpy()
#wBSMtest = wBSM_test["w"].to_numpy()
weight_train = tf.squeeze(weight_train)
#weight_train = weight_train["w"].to_numpy()
# scale data

t = MinMaxScaler()
#t = StandardScaler()
t.fit(SM_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
SM_train = t.transform(SM_train)
SM_test = t.transform(SM_test)
BSM_test = t.transform(BSM_test)

n_inputs = npd.shape[1]
original_dim = n_inputs

intermediate_dim = 100 #50 by default
input_dim = 50 #was 20 in default
half_input = 7 #was 20 in the newTest
latent_dim = 3 #was 3 for optimal performance
epochs =  2#00 #80
batchsize=128 #64

vae = VariationalAutoEncoder(original_dim,intermediate_dim,input_dim,half_input,latent_dim)  
#vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=tf.keras.losses.MeanSquaredError())
#vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),run_eagerly=True, loss="binary_crossentropy",metrics = [tf.keras.metrics.BinaryAccuracy()])
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss="binary_crossentropy",loss_weights=[10.],metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])
#vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss="binary_crossentropy",loss_weights=[10.],metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch = 8)
es = EarlyStopping(monitor='val_auc', mode='max', verbose=1,patience=20)
mc = tf.keras.callbacks.ModelCheckpoint('best_model_OSWW_and_QCD.h5', monitor='val_auc', mode='max', verbose=1, save_best_only=True)
#hist = vae.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=epochs, batch_size = batchsize, callbacks=[es,mc]) 
hist = vae.fit([SM_only_train,X_train], y_train,validation_data=([SM_only_test,X_test],y_test), epochs=epochs, batch_size = batchsize, callbacks=[es,mc]) 
#class_w = {0:1.,1:0.1}
# #hist = vae.fit(X_train, y_train,class_weight=class_w, epochs=epochs, batch_size = batchsize) 
#hist = vae.fit(SM_train, true_train, epochs=epochs, batch_size = batchsize) 

#print "new model: ", vae.summary()
#encoderDecoder =  EncoderDecoder(original_dim,intermediate_dim,input_dim,half_input,latent_dim)
#reco = encoderDecoder.predict(X_test)
outputLoss = RecoAndDKL_Loss(original_dim,intermediate_dim,input_dim,half_input,latent_dim)
recoOutput = outputLoss.predict(X_test)
encoder = LatentSpace(intermediate_dim,input_dim,half_input,latent_dim)
z = encoder.predict(X_test)

nameExtenstion = "vae_OSWW_and_QCD_DNNonReconstructedVariables_0HiddenLayer_"+str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs)+"_"+str(batchsize)
tf.keras.models.save_model(encoder,'encoder_'+nameExtenstion)
tf.keras.models.save_model(vae,'vae_'+nameExtenstion)
tf.keras.models.save_model(outputLoss,'outputLoss_'+nameExtenstion)
#numpy.savetxt("lossVAE_test_newModelDimenstions_MinMaxScaler_"+nameExtenstion+".csv",hist.history["loss"],delimiter=",")
#vae=tf.keras.models.load_model('vae_test_newModelUsingLatentSpace_'+nameExtenstion)





# outputs

vae.load_weights("best_model_OSWW_and_QCD.h5")

output_SM = vae.predict([SM_test,SM_test])
output_BSM = vae.predict([BSM_test,BSM_test])


#print output_SM
#print output_BSM
bins=100
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(output_SM,bins=bins, density=1,range=[0.,1.],histtype="step",color="red",alpha=0.6,linewidth=2,label="SM Output")                        
ax.hist(output_BSM,bins=bins, density=1,range=[0.,1.],histtype="step",color="blue",alpha=0.6,linewidth=2,label="BSM Output")                        
#plt.rc('legend',fontsize='small')    
ax.legend()

x_bins = range(len(hist.history["loss"]))
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(x_bins,hist.history["loss"],label="training losss")  
ax.plot(x_bins,hist.history["val_loss"],label="val_loss")
ax.legend()

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(x_bins,hist.history["binary_accuracy"],label="accuracy")
ax.plot(x_bins,hist.history["val_binary_accuracy"],label="val_accuracy")     
ax.legend()       

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(x_bins,hist.history["auc"],label="auc")
ax.plot(x_bins,hist.history["val_auc"],label="val_auc")     
ax.legend()       
plt.show() 
