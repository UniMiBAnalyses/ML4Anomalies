# VAE models
This folder contains the trainable VAE models.  
The following scheme represents the model **VAE_model_extended.py** and it represents the architecture that gave the best performance so far.
![Alt Text](https://github.com/GiuliaLavizzari/ML4Anomalies/blob/50f243532b371ced4eab8769c03343e20e59d69e/VAEmodel/VAE_ML4Anomalies.png)

## VAE_model_extended.py
This VAE model is built via subclassing (a guide for building models via subclassing in TensorFlow can be found [here](https://www.tensorflow.org/guide/keras/custom_layers_and_models)). The encoder and decoder are set as two separate objects, each of which inherits from the tf.keras.layers.Layer class. The two of them are put together to create the variational autoencoder, which inherits from the tf.keras.Model class. Encoder and decoder in turn comprise several layers, the dimension of which can be set by changing the parameters of the model during the training. Each layer employs a LeakyReLU as an activation function. Note that the input data (and thus the outputs of the VAE) are not bound between 0 and 1, thus the activation function of the last layer of the decoder (which direcly gives the output of the model) shouldn't be bound between 0 and 1. See the Training folder for more information on the scaling of the data. 


## VAE_model.py

## VAE_new_model.py
Encoder and decoder together (you need this to run some of the shap stuff)

## test_AE.py

## test_VAE.py

## test_VAE_class.py

## test_VAE_weights.py

## test_loadVAE.py
