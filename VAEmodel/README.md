# VAE models
This folder contains the trainable VAE models.  
The following scheme represents the model **VAE_model_extended.py** and it represents the architecture that gave the best performance so far.
![Alt Text](https://github.com/GiuliaLavizzari/ML4Anomalies/blob/50f243532b371ced4eab8769c03343e20e59d69e/VAEmodel/VAE_ML4Anomalies.png)

## VAE_model_extended.py
This VAE model is built via subclassing (a guide for building models via subclassing in TensorFlow can be found [here](https://www.tensorflow.org/guide/keras/custom_layers_and_models)).  

**Structure of the model:**  
The encoder and decoder are set as two separate objects, each of which inherits from the tf.keras.layers.Layer class. The two of them are put together to create the variational autoencoder, which inherits from the tf.keras.Model class. Encoder and decoder in turn comprise several layers, the dimension of which can be set by changing the parameters of the model during the training. Each layer employs a LeakyReLU as an activation function. Note that the input data (and thus the outputs of the VAE) are not bound between 0 and 1, thus the activation function of the last layer of the decoder (which direcly gives the output of the model) shouldn't be bound between 0 and 1. See the Training folder for more information on the scaling of the data.  
The VAE model also comprises a sampling layer; given the mean and variance produced by the encoder, this layer allows for sampling from a gaussian distribution with this mean and variance. Once the sampling operation is performed, the sampled point can be decoded. Note that the actual sampling operation is performed on a standard normal distribution which is then multiplied by parameters that are functions of the mean and variance computed. This allows for backpropagation through the sampling layer (reparametrization trick).
```python
class sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the variables."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```


**Loss function:**  
The model is trained by means of the Adam optimizer. The loss function considered for the training comprises a reconstruction term (tf.keras.losses.MeanSquaredError()) and a regularization term (Kullback-Leibler Divergence), which is added by means of the add_loss method.
```python
# training.py
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=tf.keras.losses.MeanSquaredError())

# vae_model_extended.py 
# within the definition of the class VariationalAutoEncoder
kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
kl_loss= kl_loss/1000000.
self.add_loss(kl_loss)
```
Note that the KLD is multiplied by a scaling factor << 1, which means that reconstruction is favoured over regularization during the training. This factor was chosen as the one that gave the smallest value of the total training loss.




## VAE_model.py

## VAE_new_model.py
Encoder and decoder together (you need this to run some of the shap stuff)

## test_AE.py

## test_VAE.py

## test_VAE_class.py

## test_VAE_weights.py

## test_loadVAE.py
