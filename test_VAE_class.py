from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
import sys
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

ROOT.ROOT.EnableImplicitMT()

class sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the variables."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
    """Maps PD variables to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
               latent_dim=4,
               intermediate_dim=28,
               input_dim=14,
               name='encoder',
               **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)                       
        self.encoder_layer1 = layers.Dense(intermediate_dim)
        self.encoder_active_layer1 = layers.LeakyReLU(name="encoder_leakyrelu_1")
        self.encoder_layer2 = layers.Dense(input_dim)
        self.encoder_active_layer2 = layers.LeakyReLU(name="encoder_activ_layer_2")
        self.encoder_layer3 = layers.Dense(input_dim)
        self.encoder_active_layer3 = layers.LeakyReLU(name="encoder_activ_layer_3")    
        self.encoder_layer4 = layers.Dense(input_dim)
        self.encoder_active_layer4 = layers.LeakyReLU(name="encoder_activ_layer_4")    
        self.encoder_layer5 = layers.Dense(input_dim) #to be 1/2
        self.encoder_active_layer5 = layers.LeakyReLU(name="encoder_activ_layer_5")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = sampling()

    def call(self, inputs):
        xl1 = self.encoder_layer1(inputs)
        xl1_active = self.encoder_active_layer1(xl1)
        xl2 = self.encoder_layer2(xl1_active)
        xl2_active = self.encoder_active_layer2(xl2)
        xl3 = self.encoder_layer3(xl2_active)
        xl3_active = self.encoder_active_layer3(xl3)
        xl4=self.encoder_layer4(xl3_active)
        xl4_active=self.encoder_active_layer4(xl4)
        xl5=self.encoder_layer5(xl4_active)
        xl5_active=self.encoder_active_layer5(xl5)
        z_mean = self.dense_mean(xl5_active)
        z_log_var = self.dense_log_var(xl5_active)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(layers.Layer):
    """Converts z, the encoded vector, back into a readable list of variables."""

    def __init__(self,
            original_dim,
            latent_dim=4,               
               name='decoder',
               **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder_layer1 = layers.Dense(latent_dim)
        self.decoder_active_layer1 = layers.LeakyReLU(name="decoder_leakyrelu_1")
        self.decoder_layer2 = layers.Dense(original_dim)    #to be 1/2
        self.decoder_active_layer2 = layers.LeakyReLU(name="decoder_leakyrelu_2")    
        self.decoder_layer3 = layers.Dense(original_dim)
        self.decoder_active_layer3 = layers.LeakyReLU(name="decoder_leakyrelu_3")
        self.decoder_layer4 = layers.Dense(original_dim)
        self.decoder_active_layer4 = layers.LeakyReLU(name="decoder_leakyrelu_4")
        self.dense_output = layers.Dense(original_dim)
        self.decoder_active_output = layers.LeakyReLU(name="decoder_leakyrelu_output")

    def call(self, inputs):
        layer1 = self.decoder_layer1(inputs)
        active_layer1= self.decoder_active_layer1(layer1)
        layer2=self.decoder_layer2(active_layer1)
        active_layer2=self.decoder_active_layer2(layer2)
        layer3=self.decoder_layer3(active_layer2)
        active_layer3=self.decoder_active_layer3(layer3)
        layer4=self.decoder_layer3(active_layer3)
        active_layer4=self.decoder_active_layer4(layer4)
        output =self.dense_output(active_layer4)
        return self.decoder_active_output(output)

class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
               original_dim,
               intermediate_dim,
               latent_dim=4,
               name='autoencoder',
               **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim,input_dim=original_dim)
        self.decoder = Decoder(original_dim, latent_dim)

    def call(self, inputs):
        # self._set_inputs(inputs)
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        kl_loss= kl_loss/10000.
        self.add_loss(kl_loss)
        return reconstructed


class LatentSpace(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
               original_dim,
               intermediate_dim,
               latent_dim=4,
               name='latentspace',
               **kwargs):
        super(LatentSpace, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim,input_dim=original_dim)

    def call(self, inputs):
        # self._set_inputs(inputs)
        z_mean, z_log_var, z = self.encoder(inputs)
        return z


pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
df = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")


npy = df.AsNumpy(pd_variables)
wSM = df.AsNumpy("w")
npd =pd.DataFrame.from_dict(npy)
wpdSM = pd.DataFrame.from_dict(wSM)
nEntries = 400000
npd = npd.head(nEntries*2)
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
vae = VariationalAutoEncoder(original_dim, 2*original_dim, 4)  #, input_shape=(784,)
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=tf.keras.losses.MeanSquaredError())

#vae.fit([X_train, X_train,wx], epochs=100, validation_data=(X_test,X_test),use_multiprocessing=True)
#vae.fit(X_train,X_train, epochs=30,sample_weight=wx, batch_size = 32)
vae.fit(X_train,X_train, epochs=20,sample_weight=wx, batch_size = 32)
encoder = LatentSpace(original_dim, 2*original_dim, 4)

z = encoder.predict(X_train)
#print z
#vae.save('vae_denselayers_4Dim_withWeights')
tf.keras.models.save_model(encoder,'encoder_test')
tf.keras.models.save_model(vae,'vae_denselayers_4Dim_withWeights')
#encoded_data = encoder.predict(X_test)
#decoded_data = decoder.predict(encoded_data)
#results = vae.evaluate(X_test, X_test, batch_size=32,sample_weight=wxtest)
#print("test loss, test acc:", results)
