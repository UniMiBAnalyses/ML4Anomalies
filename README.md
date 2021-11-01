# ML4Anomalies
**Some useful references on the topic:**  

* On VAEs:
    * https://keras.io/examples/generative/vae/
    * https://www.tensorflow.org/guide/keras/custom_layers_and_models
    * https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
    * https://arxiv.org/abs/1811.10276
    * https://arxiv.org/pdf/1906.02691.pdf  
 
* On VAEs for anomaly detection in HEP:
    * https://arxiv.org/pdf/1312.6114.pdf

**Some additional tests for anomaly detection:**

Keras:
Generic Keras setup and playground  
In particular the folder VAE could be useful here
* https://github.com/amassiro/MyKeras

Anomaly detection for de/dx:
Some tests for anomaly detection for de/dx based analysis
* https://github.com/amassiro/ML4Anomalies


## Contents:
### VAEmodel 
Contains the VAE models used.

### Training  
Contains the script that runs the training of the VAE models.

### Plots  
Contains tools to plot input and output variables, latent variables, correlation matrix and ROC curves.

### SHAP  
Contains tools to understand which features are contributing the most to the loss of the model.

### Trained models:
vae_batch16_newModelDimenstions_MinMaxScaler_20_10_7_3_100  
vae_test_newModelDimenstions_MinMaxScaler_20_10_7_3_100    
vae_test_newModelDimenstions_MinMaxScaler_30_20_10_5_100 


