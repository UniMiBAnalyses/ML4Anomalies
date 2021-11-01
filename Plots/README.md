# Plots

## PlotVariables.py
Plots the samples (before any preprocessing operation).

## PlotTrainingLoss.py
Uploads and plots the loss per epoch computed and saved during the training, as a function of the number of the epoch.

## PlotOUTVsIN.py
Plots the input and output SM test distributions and the latent distributions. To do so, the data are splitted and scaled as in the training. The trained VAE model and the encoder are uploaded and used to produce the output and latent distributions, respectively:
```python
model = tf.keras.models.load_model('VAEmodel_name')
out = model.predict(X_test)
encoder = tf.keras.models.load_model('encoder_name')
latent = encoder.predict(X_test)
```
Note that the model is not fed the weights of the events; instead, the weights are used to rescale both the input and output distributions when the plots are produced:
```python
# input and output distributions:
axes[i][j].hist(X_test[0:,nvar],bins=500, weights = wxtest, range=[-0.1,1.2],histtype="step",color="blue",alpha=0.6,linewidth=2,label="Input")
axes[i][j].hist(out[0:,nvar],bins=500, weights = wxtest, range=[-0.1,1.2],histtype="step",color="red",alpha=0.6,linewidth=2,label="output")
```

## PlotCombinedSamples.py
Plots:
* Correlation matrix of the input variables
* Value of the loss function for the single events
* Input and output distributions (SM test sample)
* Input and output distributions (BSM sample = SM test + LIN + QUAD)
* Latent distributions (both for the SM alone and the whole BSM sample)  

It also allows for adding a gaussian noise to the input variables.  

**Combining the samples**
The SM test sample and the LIN and QUAD samples can be combined as follows:  
First, all the samples are uploaded and cuts and logarithms are applied. The LIN and QUAD samples are then merged in the All_BSM sample. The SM sample is split as usual into X_test and X_train. X_train is then used to compute the scaling factor, which is then applied to all the samples. Eventually, X_test and All_BSM are merged. The resulting dataset contains both SM and EFT contributions.



