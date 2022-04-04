# VAE and DNN
This folder contains two models (and the related scripts for training) whose architecture comprises a Variational AutoEncoder and a DNN that serves as a classifier. The aim is that of training the model not only to reconstruct a SM sample, but also for discrimination.


The following scheme represents the model **VAE_DNN_model.py**.
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
recoLoss = tf.expand_dims(recoLoss,-1)
myOutput = self.classifier(recoLoss)
```
- The value of a bidimensional loss that comprises both reconstruction (MSE) and regularization (KLD) loss, computed for each event:
```python
recoLoss = math_ops.squared_difference(reconstructed, inputs)
recoLoss = tf.keras.backend.mean(recoLoss, axis = -1) 
recoLoss = tf.expand_dims(recoLoss,-1)
newKLLoss = tf.keras.backend.mean(- 0.5 *(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1), axis = -1)
totLoss = tf.stack([recoLoss,newKLLoss],axis=1)
myOutput = self.classifier(totLoss)
```


**Loss function:**  
The model is trained by means of the Adam optimizer. The loss function considered for the training comprises a discrimination term, the Binary Cross Entropy, and the usual MSE and KLD which are added by means of the add_loss method.
```python
# VAE_DNN_training.py
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss="binary_crossentropy",metrics = [tf.keras.metrics.BinaryAccuracy()])

# VAE_DNN_model.py 
# within the definition of the class VariationalAutoEncoder

mse = tf.keras.losses.MeanSquaredError()
mseLoss = mse(inputs, reconstructed)*0.01 #was 1. by deafult        
self.add_loss(mseLoss) 

kl_loss = - 0.5 * tf.reduce_mean(
       z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
kl_loss= kl_loss/100000000. # was 1000000.
self.add_loss(kl_loss)  
```


**Training:**
Note that both the VAE and the DNN are trained on a sample which comprises both SM and EFT events.
 

## VAE_separate.py (VAE_separate_training_forOSWW.py)
The architecture of the model is similar to that of the previous one; however, this model allows for training the VAE part only on the SM sample and the DNN part on both SM and EFT events. Indeed, the aim is that of training the VAE part only for SM reconstruction and the DNN part to discriminate between SM and BSM events.

This is achieved by giving as an input to the model a list of two objects:
```python
hist = vae.fit([SM_only_train,X_train], y_train,validation_data=([SM_only_test,X_test],y_test), epochs=epochs, batch_size = batchsize, callbacks=[es,mc])
```
where the first entry of the input data (namely, SM_only_train) contains the SM sample on which the VAE is trained, while the second entry (namely, X_train) contains SM and EFT events used by the DNN for the classification. Indeed, the losses computed on SM are added to the model, to be minimized by adam, while the losses computed on BSM (SM + EFT) are passed to the classifier:

```python
# within the definition of the Variational AutoEncoder class:

#for SM only
z_mean, z_log_var, z = self.encoder(inputs[0])
reconstructed = self.decoder(z)   

#SM+BSM 
z_mean_BSM, z_log_var_BSM, z_BSM = self.encoder(inputs[1])
reconstructed_BSM = self.decoder(z_BSM)
        

# losses computed on SM are added to the model, to be minimized by adam        
mse = tf.keras.losses.MeanSquaredError()
mseLoss = mse(inputs[0], reconstructed)*1. #was 1. by deafult        
self.add_loss(mseLoss) 

kl_loss = - 0.5 * tf.reduce_mean(
       z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
kl_loss= kl_loss/1000000. # was 1000000.
self.add_loss(kl_loss)  
        
        
# losses computed on BSM are passed to the classifier
recoLoss = math_ops.squared_difference(reconstructed_BSM, inputs[1])
recoLoss = tf.keras.backend.mean(recoLoss, axis = -1)       
newKLLoss = tf.keras.backend.mean(- 0.5 *(z_log_var_BSM - tf.square(z_mean_BSM) - tf.exp(z_log_var_BSM) + 1), axis = -1)
totLoss = tf.stack([recoLoss,newKLLoss],axis=1)
myOutput = self.classifier(totLoss)       
```


Note also that during the trainig Keras Model Checkpoint and Early Stopping are used: the monitored metric based on which the training is stopped and the weights are saved is the AUC value (the model is thus chosen as the best in terms of discrimination).
```python
es = EarlyStopping(monitor='val_auc', mode='max', verbose=1,patience=20)
mc = tf.keras.callbacks.ModelCheckpoint('best_model_OSWW_and_QCD.h5', monitor='val_auc', mode='max', verbose=1, save_best_only=True)
hist = vae.fit([SM_only_train,X_train], y_train,validation_data=([SM_only_test,X_test],y_test), epochs=epochs, batch_size = batchsize, callbacks=[es,mc]) 
```
