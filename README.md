# ML4Anomalies
**Some useful references on the topic:**  

On VAEs:
* https://keras.io/examples/generative/vae/
* https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
* https://arxiv.org/abs/1811.10276
* https://arxiv.org/pdf/1906.02691.pdf  
 
On VAEs for anomaly detection in HEP:
* https://arxiv.org/pdf/1312.6114.pdf

Some additional tests for anomaly detection:

Keras:
Generic Keras setup and playground  
In particular the folder VAE could be useful here
* https://github.com/amassiro/MyKeras

Anomaly detection for de/dx:
Some tests for anomaly detection for de/dx based analysis
* https://github.com/amassiro/ML4Anomalies


## VAEmodel
Contains the VAE model and the script that runs the training.

## Plots
Contains tools to plot input and output variables, latent variables, the correlation matrix and ROC curves (which gives a measure of the efficiency of te anomaly detection).

## BSM analysis
Contains tools to perform the anomaly detection.
