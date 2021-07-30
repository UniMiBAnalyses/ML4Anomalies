from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import sys
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


#taking the model
from VAE_model_extended import *

import ROOT
#ROOT.ROOT.EnableImplicitMT()





#
# variable from the nutple
#
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
dfAll = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")


npy = df.AsNumpy(pd_variables)
wSM = df.AsNumpy("w")
npd =pd.DataFrame.from_dict(npy)
wpdSM = pd.DataFrame.from_dict(wSM)
nEntries = 1000000
npd = npd.head(nEntries*2)
#to be done for all the pt and mass and met variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    npd[vars] = npd[vars].apply(numpy.log10)
wpdSM = wpdSM.head(nEntries*2)

X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
#print wx_train,X_train
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()
# scale data
t = MinMaxScaler()
#t = StandardScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

n_inputs = npd.shape[1]
original_dim = n_inputs

intermediate_dim = 150 #50 by default
input_dim = 100 #was 20 in default
half_input = 50 #was 20 in the newTest
latent_dim = 7
epochs = 200
vae = VariationalAutoEncoder(original_dim,intermediate_dim,input_dim,half_input,latent_dim)  
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=tf.keras.losses.MeanSquaredError())

#hist = vae.fit(X_train,X_train, epochs=250,sample_weight=wx, batch_size = 16)
hist = vae.fit(X_train,X_train, epochs=epochs, batch_size = 32)

#encoder = LatentSpace(latent_dim, intermediate_dim, original_dim, half_input)

#z = encoder.predict(X_train)
#print z
#vae.save('vae_denselayers_4Dim_withWeights')
#tf.keras.models.save_model(encoder,'encoder_test_newModelDimenstions_MinMaxScaler')
tf.keras.models.save_model(vae,'vae_test_newModelDimenstions_MinMaxScaler_'+str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs))
numpy.savetxt("loss_test_newModelDimenstions_MinMaxScaler_"+str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs)+".csv", hist.history["loss"],delimiter=',')
#encoded_data = encoder.predict(X_test)
#decoded_data = decoder.predict(encoded_data)
#results = vae.evaluate(X_test, X_test, batch_size=32,sample_weight=wxtest)
#print("test loss, test acc:", results)
