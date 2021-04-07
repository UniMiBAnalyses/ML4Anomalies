from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
import sys
import numpy
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

ROOT.ROOT.EnableImplicitMT()

def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance/2) * epsilon
    return random_sample

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1/1000.
        #reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict))
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance))
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance))
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
                
        return loss
    return vae_loss

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
df = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
dfBSM = ROOT.RDataFrame("SSWW_cW_QU","../ntuple_SSWW_cW_QU.root")
dfBSM2 = ROOT.RDataFrame("SSWW_cHW_QU","../ntuple_SSWW_cHW_QU.root")

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
npdBSM2 = npdBSM2.head(nEntries)

#print npd.columns
#print npdBSM.columns

"""
for i in range(6):
    npd_discr["dau1_deepTauVsJet"] = npd_discr["dau1_deepTauVsJet"].replace([i],0)
for j in range(6,9):
    npd_discr["dau1_deepTauVsJet"] = npd_discr["dau1_deepTauVsJet"].replace([j],1)

#print npd_discr
"""

X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.33, random_state=1)
n_inputs = npd.shape[1]
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

visible = Input(shape=(n_inputs,))
# encoder level 1
encoder_conv_layer1 = Dense(n_inputs*2)(visible)
#Conv1D(filters=n_inputs, kernel_size = 1,name="encoder_conv_1")(visible) #Need to understand the dimensionality ...
encoder_norm_layer1 = BatchNormalization(name="encoder_norm_1")(encoder_conv_layer1)
encoder_activ_layer1 = LeakyReLU(name="encoder_leakyrelu_1")(encoder_norm_layer1)
# encoder level 2, 3,4,5
encoder_conv_layer2 = Dense(n_inputs)(encoder_activ_layer1)
#Conv1D(filters=32, kernel_size =1,name="encoder_conv_2")(encoder_activ_layer1)
encoder_norm_layer2 = BatchNormalization(name="encoder_norm_2")(encoder_conv_layer2)
encoder_activ_layer2 = LeakyReLU(name="encoder_activ_layer_2")(encoder_norm_layer2)

encoder_conv_layer3 = Dense(n_inputs)(encoder_activ_layer2)
#Conv1D(filters=64, kernel_size =1,name="encoder_conv_3")(encoder_activ_layer2)
encoder_norm_layer3 = BatchNormalization(name="encoder_norm_3")(encoder_conv_layer3)
encoder_activ_layer3 = LeakyReLU(name="encoder_activ_layer_3")(encoder_norm_layer3)

encoder_conv_layer4 = Dense(n_inputs)(encoder_activ_layer3)
#Conv1D(filters=64, kernel_size =1,name="encoder_conv_4")(encoder_activ_layer3)
encoder_norm_layer4 = BatchNormalization(name="encoder_norm_4")(encoder_conv_layer4)
encoder_activ_layer4 = LeakyReLU(name="encoder_activ_layer_4")(encoder_norm_layer4)

encoder_conv_layer5 = Dense(round(float(n_inputs)/2.))(encoder_activ_layer4)
#Conv1D(filters=64, kernel_size =1, name="encoder_conv_5")(encoder_activ_layer4)
encoder_norm_layer5 = BatchNormalization(name="encoder_norm_5")(encoder_conv_layer5)
encoder_activ_layer5 = LeakyReLU(name="encoder_activ_layer_5")(encoder_norm_layer5)
#Do we need flattening?
shape_before_flatten = tf.keras.backend.int_shape(encoder_activ_layer5)[1:]
encoder_flatten = tf.keras.layers.Flatten()(encoder_activ_layer5)

#Up to here we could have used the AE of the other example ....
#now we make the "variational part", i.e. we create the Normal distribution mu and sigma:
#latent_space_dim = round(float(n_inputs)/3.)
latent_space_dim = 2
encoder_mu = Dense(units=latent_space_dim, name="encoder_mu")(encoder_flatten)
encoder_log_variance = Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_flatten)

#this is the sampling part
encoder_output = tf.keras.layers.Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])

#now let's init the model for the encoder:
encoder = tf.keras.models.Model(visible, encoder_output, name="encoder_model")

#sencoder.summary()

decoder_input = Input(shape=(latent_space_dim), name="decoder_input")
#This is to restore the proper shape pf the input before the latent space in the encoder
decoder_dense_layer1 = Dense(units=numpy.prod(shape_before_flatten), name="decoder_dense_1")(decoder_input)
#do we need this if we do not use conv1D or conv2D?
decoder_reshape = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(decoder_dense_layer1)

#Now making the decoder steps
decoder_conv_tran_layer1 = Dense(round(float(n_inputs)/2.))(decoder_reshape)
decoder_norm_layer1 = BatchNormalization(name="decoder_norm_1")(decoder_conv_tran_layer1)
decoder_activ_layer1 = LeakyReLU(name="decoder_leakyrelu_1")(decoder_norm_layer1)

decoder_conv_tran_layer2 = Dense(n_inputs)(decoder_activ_layer1)
decoder_norm_layer2 = BatchNormalization(name="decoder_norm_2")(decoder_conv_tran_layer2)
decoder_activ_layer2 = LeakyReLU(name="decoder_leakyrelu_2")(decoder_norm_layer2)

decoder_conv_tran_layer3 =  Dense(n_inputs)(decoder_activ_layer2)
decoder_norm_layer3 = BatchNormalization(name="decoder_norm_3")(decoder_conv_tran_layer3)
decoder_activ_layer3 = LeakyReLU(name="decoder_leakyrelu_3")(decoder_norm_layer3)

decoder_conv_tran_layer4 = Dense(n_inputs)(decoder_activ_layer3)
decoder_output = LeakyReLU(name="decoder_output")(decoder_conv_tran_layer4 )

decoder = Model(decoder_input, decoder_output, name="decoder_model")

#decoder.summary()

vae_encoder_output = encoder(visible)
vae_decoder_output = decoder(vae_encoder_output)
vae = tf.keras.models.Model(visible, vae_decoder_output, name="VAE")
#The following line has to be re-done with the proper loss_func
#vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss="mse")
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=loss_func(encoder_mu, encoder_log_variance))


#vae.fit(X_train, X_train, epochs=100, batch_size=16, shuffle=True, validation_data=(X_test, X_test))
vae.fit(X_train, X_train, epochs=20, batch_size=16, validation_data=(X_test, X_test))
encoded_data = encoder.predict(X_test)
decoded_data = decoder.predict(encoded_data)
results = vae.evaluate(X_test, X_test, batch_size=128)
print("test loss, test acc:", results)
#t.fit(npdBSM)
npdBSM = t.transform(npdBSM)
results = vae.evaluate(npdBSM, npdBSM, batch_size=128)
print("test loss, test acc:", results)
#t.fit(npdBSM2)
npdBSM2 = t.transform(npdBSM2)
results = vae.evaluate(npdBSM2, npdBSM2, batch_size=128)
print("BSM2 loss, BSM2 acc:", results)
