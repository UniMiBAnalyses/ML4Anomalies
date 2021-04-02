from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
#import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

ROOT.ROOT.EnableImplicitMT()


pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll', 'phij1', 'phij2', 'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll', 'w']
df = ROOT.RDataFrame("SSWW_SM","./ntuple_SSWW_SM.root")
dfBSM = ROOT.RDataFrame("SSWW_cW_QU","./ntuple_SSWW_cW_QU.root")
dfBSM2 = ROOT.RDataFrame("SSWW_cHW_QU","ntuple_SSWW_cHW_QU.root")

npy = df.AsNumpy(pd_variables)
npd =pd.DataFrame.from_dict(npy)
#print npd

#npd = npd[(npd["ptj1"] > 200)]

npyBSM = dfBSM.AsNumpy(pd_variables)
npdBSM =pd.DataFrame.from_dict(npyBSM)

npyBSM2 = dfBSM2.AsNumpy(pd_variables)
npdBSM2 =pd.DataFrame.from_dict(npyBSM2)
#print npd

#Just reducing a bit the sample
#npd = npd[(npd["ptj1"] > 200)]
#npdBSM = npdBSM[(npdBSM["ptj1"] > 200)]
nEntries = 200000
npd = npd.head(nEntries)
npdBSM = npdBSM.head(nEntries)
npdBSM2 = npdBSM.head(nEntries)

print npd.columns
#print npdBSM.columns

"""
for i in range(6):
    npd_discr["dau1_deepTauVsJet"] = npd_discr["dau1_deepTauVsJet"].replace([i],0)
for j in range(6,9):
    npd_discr["dau1_deepTauVsJet"] = npd_discr["dau1_deepTauVsJet"].replace([j],1)

#print npd_discr
"""

#X_train, X_test, y_train, y_test = train_test_split(npd, npd["mll"], test_size=0.33, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.33, random_state=1)
n_inputs = npd.shape[1]
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
visible = Input(shape=(n_inputs,))


# encoder level 1
e = Dense(n_inputs*2)(visible)
e = Dropout(0.2)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = Dropout(0.2)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = n_inputs
n_bottleneck = round(float(n_inputs) / 1.0)
bottleneck = Dense(n_bottleneck)(e)

# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
e = Dropout(0.2)(e)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs*2)(d)
e = Dropout(0.2)(e)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)

# define autoencoder model
model = Model(inputs=visible, outputs=output)
model.summary()
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
# plot the autoencoder
plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
# fit the autoencoder model to reconstruct input
history = model.fit(X_train, X_train, epochs=100, batch_size=16, verbose=1, validation_data=(X_test,X_test))
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
#pyplot.show()
# define an encoder model (without the decoder)
#encoder = Model(inputs=visible, outputs=bottleneck)
#plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
# save the encoder to file
model.save('autoencoder.h5')

results = model.evaluate(X_test, X_test, batch_size=128)
print("test loss, test acc:", results)

t.fit(npdBSM)
npdBSM = t.transform(npdBSM)
results = model.evaluate(npdBSM, npdBSM, batch_size=128)
print("test loss, test acc:", results)

npdBSM2 = t.transform(npdBSM2)
results = model.evaluate(npdBSM2, npdBSM2, batch_size=128)
print("test loss, test acc:", results)
