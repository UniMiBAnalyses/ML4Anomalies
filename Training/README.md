# Training the model

## training.py 
Trains the model and saves the encoder and the VAE model, together with a .csv file containing the values of the losses per epoch.  

Importing the model:
```python
from VAEmodel import * # where the chosen model is VAEmodel.py
```
Saving the data and applying cuts to some variables:
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
Splitting the data into train and test dataset:
```python
X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()
```
Preprocessing:
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

Training the model:
```python
# whole model
vae = VariationalAutoEncoder(original_dim, DIM) # (self, original, latent)
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss=tf.keras.losses.MeanSquaredError())
hist = vae.fit(X_train, X_train, epochs=EPOCHS, batch_size = BATCH)
```
```python
# encoder
encoder = LatentSpace(original_dim, DIM) # (self, original, latent)
z = encoder.predict(X_train)
```
Saving the model:
```python
enc_name = "myenc{}_denselayers_latentdim{}_epoch{}_batchsize{}_log_eventFiltered".format(MODEL, DIM, EPOCHS, BATCH)
vae_name = "myvae{}_denselayers_latentdim{}_epoch{}_batchsize{}_log_eventFiltered".format(MODEL, DIM, EPOCHS, BATCH)
csv_name = "myloss{}_training_latentdim{}_epoch{}_batchsize{}_log_eventFiltered.csv".format(MODEL, DIM, EPOCHS, BATCH)
tf.keras.models.save_model(encoder, enc_name) 
tf.keras.models.save_model(vae, vae_name)
np.savetxt(csv_name, hist.history["loss"], delimiter=',')
```
