import tensorflow as tf
from tensorflow.keras import layers

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
               half_input=7,
               name='encoder',
               **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)                       
        self.encoder_layer1 = layers.Dense(intermediate_dim)        
        self.encoder_active_layer1 = layers.LeakyReLU(name="encoder_leakyrelu_1")
        self.encoder_layer2 = layers.Dense(input_dim)
        #self.encoder_norm_layer2 = layers.BatchNormalization(name="encoder_norm_2")
        self.encoder_active_layer2 = layers.LeakyReLU(name="encoder_activ_layer_2")
        self.encoder_layer3 = layers.Dense(input_dim)
        #self.encoder_norm_layer3 = layers.BatchNormalization(name="encoder_norm_3")
        self.encoder_active_layer3 = layers.LeakyReLU(name="encoder_activ_layer_3")    
        self.encoder_layer4 = layers.Dense(input_dim)
        #self.encoder_norm_layer4 = layers.BatchNormalization(name="encoder_norm_4")
        self.encoder_active_layer4 = layers.LeakyReLU(name="encoder_activ_layer_4")    
        self.encoder_layer5 = layers.Dense(input_dim) #to be 1/2
        self.encoder_layer5 = layers.Dense(half_input) 
        #self.encoder_norm_layer5 = layers.BatchNormalization(name="encoder_norm_5")
        self.encoder_active_layer5 = layers.LeakyReLU(name="encoder_activ_layer_5")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = sampling()

    def call(self, inputs):
        xl1 = self.encoder_layer1(inputs)
        xl1_active = self.encoder_active_layer1(xl1)        
        xl2 = self.encoder_layer2(xl1_active)
        #xl2_norm = self.encoder_norm_layer2(xl2)
        xl2_active = self.encoder_active_layer2(xl2)
        xl3 = self.encoder_layer3(xl2_active)
        #xl3_norm = self.encoder_norm_layer3(xl3)
        xl3_active = self.encoder_active_layer3(xl3)
        xl4=self.encoder_layer4(xl3_active)
        #xl4_norm = self.encoder_norm_layer4(xl4)
        xl4_active=self.encoder_active_layer4(xl4)
        xl5=self.encoder_layer5(xl4_active)
        #xl5_norm = self.encoder_norm_layer5(xl5)
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
            half_input=7,             
               name='decoder',
               **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder_layer1 = layers.Dense(latent_dim)
        #self.decoder_norm_layer1 = layers.BatchNormalization(name="decoder_norm_1")
        self.decoder_active_layer1 = layers.LeakyReLU(name="decoder_leakyrelu_1")
        self.decoder_layer2 = layers.Dense(original_dim)    #to be 1/2        
        #self.decoder_norm_layer2 = layers.BatchNormalization(name="decoder_norm_2")
        self.decoder_active_layer2 = layers.LeakyReLU(name="decoder_leakyrelu_2")    
        self.decoder_layer3 = layers.Dense(original_dim)
        #self.decoder_norm_layer3 = layers.BatchNormalization(name="decoder_norm_3")
        self.decoder_active_layer3 = layers.LeakyReLU(name="decoder_leakyrelu_3")
        #self.decoder_layer4 = layers.Dense(original_dim)
        self.decoder_layer4 = layers.Dense(half_input)
        #self.decoder_norm_layer4 = layers.BatchNormalization(name="decoder_norm_4")
        self.decoder_active_layer4 = layers.LeakyReLU(name="decoder_leakyrelu_4")
        self.dense_output = layers.Dense(original_dim)
        self.decoder_active_output = layers.LeakyReLU(name="decoder_leakyrelu_output")

    def call(self, inputs):
        layer1 = self.decoder_layer1(inputs)
        #layer1_norm = self.decoder_norm_layer1(layer1)
        active_layer1= self.decoder_active_layer1(layer1)
        layer2=self.decoder_layer2(active_layer1)
        #layer2_norm = self.decoder_norm_layer2(layer2)
        active_layer2=self.decoder_active_layer2(layer2)
        layer3=self.decoder_layer3(active_layer2)
        #layer3_norm = self.decoder_norm_layer3(layer3)
        active_layer3=self.decoder_active_layer3(layer3)
        layer4=self.decoder_layer4(active_layer3)
        #layer4_norm = self.decoder_norm_layer4(layer4)
        active_layer4=self.decoder_active_layer4(layer4)
        output =self.dense_output(active_layer4)
        return self.decoder_active_output(output)

class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
               original_dim,
               intermediate_dim,
               latent_dim=4,
               half_input=7,
               name='autoencoder',
               **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim,input_dim=original_dim,half_input=half_input)
        self.decoder = Decoder(original_dim, latent_dim,half_input=half_input)

    def call(self, inputs):
        # self._set_inputs(inputs)
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        kl_loss= kl_loss/1000000.
        self.add_loss(kl_loss)
        return reconstructed

class LatentSpace(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
               original_dim,
               intermediate_dim,
               latent_dim=4,
               half_input=7,
               name='latentspace',
               **kwargs):
        super(LatentSpace, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim,input_dim=original_dim,half_input=half_input)

    def call(self, inputs):
        # self._set_inputs(inputs)
        z_mean, z_log_var, z = self.encoder(inputs)
        return z