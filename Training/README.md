# Training the model

## training.py 
Trains the model and saves the trained encoder and the VAE model, together with a .csv file containing the values of the losses per epoch.  

**Importing the model:**
```python
from VAEmodel import * # where the chosen model is VAEmodel.py
```

**Uploading the data samples and applying cuts on some of the variables:**
```python
# selecting the variables used for the training
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
dfAll = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")

# cuts
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200") 

npy = df.AsNumpy(pd_variables)
npd =pd.DataFrame.from_dict(npy)

# storing the weights of the events
wSM = df.AsNumpy("w")
wpdSM = pd.DataFrame.from_dict(wSM)
```

**Splitting the data into train and test datasets:**
```python
X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()
```

**Preprocessing:**
```python
#logarithm of the kinematic variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1', 'ptl2', 'ptll']:
	npd[vars] = npd[vars].apply(np.log10)

#scaling the data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
```
MinMaxScaler() scales the data within the range [0,1]. Note that the same scaling will be applied on all the data samples: X_train, X_test and the LIN and QUAD samples. This means that in all cases except X_train the values will outrange the [0,1] interval by a little. This has consequences in terms of the choice of the activation function of the last layer of the decoder, which directly determines the output (see VAE_model_extended.py).


**Setting the parameters of the model:**  
This allows to define the dimensions of the layers in the model.
```python
intermediate_dim = 20 #50 by default
input_dim = 10 #was 20 in default
half_input = 7 #was 20 in the newTest
latent_dim = 3
epochs = 50
```

**Defining and training the model:**
```python
# whole model
vae = VariationalAutoEncoder(original_dim,intermediate_dim,input_dim,half_input,latent_dim)  
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=tf.keras.losses.MeanSquaredError())
hist = vae.fit(X_train,X_train, epochs=epochs, batch_size = 32)
```
```python
# encoder
encoder = LatentSpace(intermediate_dim,input_dim,half_input,latent_dim)
z = encoder.predict(X_train)
```

**Saving the trained model:**
```python
tf.keras.models.save_model(encoder,'latent_test_newModelDimenstions_MinMaxScaler_'+str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs))
tf.keras.models.save_model(vae,'vae_test_newModelDimenstions_MinMaxScaler_'+str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs))
numpy.savetxt("lossAE_test_newModelDimenstions_MinMaxScaler_"+str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs)+".csv", hist.history["loss"],delimiter=',')
```
