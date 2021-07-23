from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy
import pandas as pd
import tensorflow as tf

#taking the model
from VAE_model import *

import ROOT

#
# variable from the nutple
#
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
dfAll = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")


npy = df.AsNumpy(pd_variables)
npd =pd.DataFrame.from_dict(npy)
nEntries = 1000000
npd = npd.head(nEntries*2)
#to be done for all the pt and mass and met variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    npd[vars] = npd[vars].apply(numpy.log10)

X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)

# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

n_inputs = npd.shape[1]
original_dim = n_inputs
latent_dim = 2
intermediate_dim = 14
numberOfEpochs = 2
vae = VariationalAutoEncoder(original_dim, 2*original_dim, latent_dim,intermediate_dim)  
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=tf.keras.losses.MeanSquaredError())
hist = vae.fit(X_train,X_train, epochs=numberOfEpochs, batch_size = 32, validation_data=(X_test, X_test))

tf.keras.models.save_model(vae,'test_vaemodel_school'+str(numberOfEpochs)+"Epochs")
numpy.savetxt("test_vaemodel_school_loss"+str(numberOfEpochs)+"Epochs.csv", hist.history["loss"],delimiter=',')

encoder = LatentSpace(original_dim, 2*original_dim, latent_dim, intermediate_dim)
encoder.predict(X_test)
tf.keras.models.save_model(encoder,'test_encoder_school'+str(numberOfEpochs)+"Epochs")