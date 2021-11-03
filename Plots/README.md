# Plots
## plotVariables.py
Plots the samples (before any preprocessing operation).

## plotTrainingLoss.py
Uploads and plots the loss per epoch computed and saved during the training, as a function of the number of the epoch.

## plotOUTVsIN.py
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

## plotCombinedSamples.py
Plots:
* Correlation matrix of the input variables
* Value of the loss function for the single events
* Input and output distributions (SM test sample)
* Input and output distributions (BSM sample = SM test + LIN + QUAD)
* Difference between input and output distributions
* Latent distributions (both for the SM alone and the whole BSM sample)  

It also allows for adding a gaussian noise to the input variables.  

### Noise
The smearing of the distributions is obtained by adding to each variable value of each event a noise factor, which is sampled from a gaussian distribution with zero mean and variance equal to the value of the variable itself, multiplied by 0.1.
```python
mu, sigma =  0 , 0.1
for var in pd_variables:
    if var != "w":
        for i in range(len(w_test)):
            sigmaBSM = sigma * np.abs(All_test[i,nvar])                    
            noise = np.random.normal(mu, sigmaBSM, 1)
            All_test[i,nvar]=All_test[i,nvar]+noise
    nvar=nvar+1
```

### Combining the samples  
The SM test sample and the LIN and QUAD samples can be combined as follows:  
First, all the samples are uploaded and cuts and logarithms are applied. The LIN and QUAD samples are then merged in the All_BSM sample. The SM sample is split as usual into X_test and X_train. X_train is then used to compute the scaling factor, which is then applied to all the samples. Eventually, X_test and All_BSM are merged. The resulting dataset contains both SM and EFT contributions. 
```python
All_BSM = pd.concat([BSM_quad, BSM_lin], keys=['Q','L'])
X_train, X_test, y_train, y_test = train_test_split(SM,SM,test_size=0.2, random_state=1)
All_test = np.concatenate((All_BSM, X_test))
```
**Weights:** Note that the weights of the EFT samples need to be scaled by the wilson coefficient cW:
```python
All_BSM["w"].loc["L"] = All_BSM["w"].loc["L"].to_numpy()*cW
All_BSM["w"].loc["Q"] = All_BSM["w"].loc["Q"].to_numpy()*cW*cW
```


### loss function
The loss function is computed by means of the LossPerBatch class, which inherits from the tf.keras.callbacks.Callback class. By choosing the batch size equal to 1, one can compute the value of the loss function on the single event of the given sample.
```python
mylosses = LossPerBatch()
model.evaluate(X_test,X_test,batch_size=1,callbacks=[mylosses],verbose=0)
myloss = mylosses.eval_loss
myloss =np.asarray(myloss)
```


## plotVariabiliLossTail.py
Plots variables that show high reconstruction error: indeed, the Mean Squared Error between input and output is computed. The events that show an MSE value which is larger than a threshold value are selected and plotted


## lossCutROC.py
Given the weights and losses per event computed by plotCombinedSamples.py, it computes and plots the ROC curves (the SM efficiency vs BSM efficiency) and plots the loss function.  
The efficiencies are computed as:  
* effSM = sumWSM*/sumWSM
* effBSM = sumWBSM*/sumWBSM  

where *sum* stands for the sum of the weights of all the SM (BSM) events, while *sum* * stands for sum of the weights of the BSM=SM+EFT events whose loss exceeds a chosen threshold value (cut)


## plotSigma.py
This script employs the losses and weights computed by means of the loss_per_batch class and plots the loss function, computes the significance sigma and the minimum value of the wilson coefficient for which the VAE model is sensitive to the EFT operator.  

First, the weights need to be correctly normalized (note that the weights taken as an input are expected to be neither scaled by a normalization factor nor scaled by the wilson coefficients yet). The correct normalization of the events is given by the following factor, that is to be multiplied to the weights of the events (together with the cW and cW^2 coefficients in the cases of the EFT events).  
```python
# Normalization factor: SM sample
luminosity = 1000.*350. #luminosity expected in 1/pb
fSM = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
hSM = fSM.Get("SSWW_SM_nums")
xsecSM = hSM.GetBinContent(1)
sumwSM = hSM.GetBinContent(2)
normSM = 5.* xsecSM * luminosity / (sumwSM) # on test set (0.2*total)

# Normalization factor: LIN sample
fLIN = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuplesBSM/ntuple_SSWW_"+str(op)+"_LI.root")
hLIN = fLIN.Get("SSWW_"+str(op)+"_LI_nums")
xsecLIN = hLIN.GetBinContent(1)
sumwLIN = hLIN.GetBinContent(2)
normLIN = xsecLIN * luminosity / (sumwLIN)

# Normalization factor: QUAD sample
fQUAD = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuplesBSM/ntuple_SSWW_"+str(op)+"_QU.root")
hQUAD = fQUAD.Get("SSWW_"+str(op)+"_QU_nums")
xsecQUAD = hQUAD.GetBinContent(1)
sumwQUAD = hQUAD.GetBinContent(2)
normQUAD = xsecQUAD * luminosity / (sumwQUAD)
```

The **significance** sigma is computed as the number of EFT (LIN + QUAD) events whose loss is bigger than a selected threshold, divided by the square root of the number of SM events above the same threshold. This is the chosen figure of merit to compare the sensitivity of different VAE models to a particular operator.  
The error on sigma due to the fluctuations of the number of events in the Monte Carlo samples is also computed.  

It's also possible to compute the **minimum value of the wilson coefficient** for which the analysis is sensitive to the EFT operator (where sensitivity is defined as having at least one value of sigma>3).
