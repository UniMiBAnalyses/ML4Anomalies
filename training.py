from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
import sys
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

#taking the model
from VAE_model import *

ROOT.ROOT.EnableImplicitMT()





#
# variable from the nutple
#
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
df = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")


npy = df.AsNumpy(pd_variables)
wSM = df.AsNumpy("w")
npd =pd.DataFrame.from_dict(npy)
wpdSM = pd.DataFrame.from_dict(wSM)
nEntries = 1000000
npd = npd.head(nEntries*2)
#to be done for all the pt and mass and met variables
#for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
#       'ptl2', 'ptll']:
#    npd[vars] = npd[vars].apply(numpy.log10)
wpdSM = wpdSM.head(nEntries*2)

X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
#print wx_train,X_train
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

n_inputs = npd.shape[1]
original_dim = n_inputs
latent_dim = 6
intermediate_dim = 7
vae = VariationalAutoEncoder(original_dim, 2*original_dim, latent_dim,intermediate_dim)  
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=tf.keras.losses.MeanSquaredError())


#vae.fit([X_train, X_train,wx], epochs=100, validation_data=(X_test,X_test),use_multiprocessing=True)
#vae.fit(X_train,X_train, epochs=30,sample_weight=wx, batch_size = 32)
hist = vae.fit(X_train,X_train, epochs=200,sample_weight=wx, batch_size = 32)
numpy.savetxt("loss_training.csv", hist.history["loss"],delimiter=',')
encoder = LatentSpace(original_dim, 2*original_dim, latent_dim, intermediate_dim)

z = encoder.predict(X_train)
#print z
#vae.save('vae_denselayers_4Dim_withWeights')
tf.keras.models.save_model(encoder,'encoder_6_latentDim_BatchNorm')
tf.keras.models.save_model(vae,'vae_denselayers_withWeights_6_latentDim_100epoch_BatchNorm')
#encoded_data = encoder.predict(X_test)
#decoded_data = decoder.predict(encoded_data)
#results = vae.evaluate(X_test, X_test, batch_size=32,sample_weight=wxtest)
#print("test loss, test acc:", results)
