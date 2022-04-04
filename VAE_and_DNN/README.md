# VAE and DNN
This folder contains two models (and the related scripts for training) whose architecture comprises a Variational AutoEncoder and a DNN which serves as a classifier. The aim is that of training the model not only to reconstruct a SM sample, but also for discrimination.


The following scheme represents the model **VAE_and_DNN.py**.
![Alt Text](https://github.com/GiuliaLavizzari/ML4Anomalies/blob/newdocu/VAE_and_DNN/VAE_semisupervised_model.png)

## VAE_DNN_model.py (VAE_DNN_training.py)
This VAE model is built via subclassing. The model comprises a simple VAE, made of an Encoder and a Decoder just as the models used so far, and a DNN that serves as a classifier. As encoder and decoder, the classifier is set as a separate object which inherits from the tf.keras.layers.Layer class. When encoder, decoder, and classifier are combined into the end-to-end model for training, the RECO and KLD losses are computed and can be given as inputs to the classifier. The training of the classifier happens through the minimization of a Binary Cross Entropy loss function.

**Classifier**
The classifier can take three different inputs, based on which it discriminates between SM and BSM events:
- The input data:
```python
myOutput = self.classifier(z)
```
- The value of the reconstruction loss (MSE) between input and output of the VAE part, computed for each event:
```python
recoLoss = math_ops.squared_difference(reconstructed, inputs)
recoLoss = tf.keras.backend.mean(recoLoss, axis = -1) 
dim_batch =  reconstructed.get_shape().as_list()[0]   
recoLoss = tf.expand_dims(recoLoss,-1)
myOutput = self.classifier(recoLoss)
```
- The value of a bidimensional loss that comprises both reconstruction (MSE) and regularization (KLD) loss, computed for each event:
```python
recoLoss = math_ops.squared_difference(reconstructed, inputs)
recoLoss = tf.keras.backend.mean(recoLoss, axis = -1) 
dim_batch =  reconstructed.get_shape().as_list()[0]   
recoLoss = tf.expand_dims(recoLoss,-1)
newKLLoss = tf.keras.backend.mean(- 0.5 *(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1), axis = -1)
totLoss = tf.stack([recoLoss,newKLLoss],axis=1)
myOutput = self.classifier(totLoss)
```

**Structure of the model:**  
The encoder and decoder are set as two separate objects, each of which inherits from the tf.keras.layers.Layer class. The two of them are put together to create the variational autoencoder, which inherits from the tf.keras.Model class. Encoder and decoder in turn comprise several layers, the dimension of which can be set by changing the parameters of the model during the training. Each layer employs a LeakyReLU as an activation function. Note that the input data (and thus the outputs of the VAE) are not bound between 0 and 1, thus the activation function of the last layer of the decoder (which direcly gives the output of the model) shouldn't be bound between 0 and 1. See the Training folder for more information on the scaling of the data.  
The definition of encoder and decoder as separate objects also allows for accessing the variables in the latent space: by saving the trained encoder alone one can indeed retrieve the latent distributions.

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

## VAE_model2.py
This model is built as the previous one. However, the structure of encoder and decoder is changed: the number and size of the layers is different than in the previous case.

## VAE_model3.py
Another possible model structure. Besides having different number and size of layers (128-64-32-latent-32-64-128), this model also employs a weight initializer (tf.keras.initializers.he_normal()) and uses a linear function as the activation function on the last layer of the decoder.

## VAE_new_model.py
In this model, encoder and decoder are not defined as separate objects: indeed, the Variational AutoEncoder class (tf.keras.Model) comprises several layers including those that form the encoder and those forming the decoder.  
This allows for a straightforward access to the layers ot the model (e.g. by means of the model.get_layer method - see testSHAP4AE_new.py for an example of its use -, that would otherwise only allow to access either the encoder or the decoder, which in VAE_model_extended.py are set as layer objects themselves).

## VAE_model_losses.py
This model allows for keeping track of the values of the losses (MSE, KLD and total) per epoch during the training. To retrieve these values, the following lines should be added to the script that runs the training:
```python
np.savetxt(loss_name, hist.history["loss"], delimiter=',')
np.savetxt(mse_name, hist.history['reconstruction_loss'], delimiter=',')
np.savetxt(kld_name, hist.history['kl_loss'], delimiter=',')
```
