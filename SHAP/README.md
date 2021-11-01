# SHAP
SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the impact of each feature.  
For more information on SHAP: 
* https://arxiv.org/pdf/1705.07874.pdf
* https://shap.readthedocs.io/en/latest/

## variablesHighLoss.py 
Shows which variables contribute the most to the recontruction loss of the events.  
First, for every event it computes the error between input and output as the absolute value of the difference input-output.  
Then, for those events whose error is greater then a selected threshold value, it computes the reconstruction error for every variable and then sorts those values: thus, for every event it's possible to identify for example the 5 variables that impact the recontruction error the most.  
After this process has been run over all the events, it counts the frequency with which each variable appeared: this gives an idea of the variables that contribute the most to the global recontruction error.

## testSHAP4AE_allEntries.py
Plots the shap summary plot (more information on shap.summary_plot() can be found [here](https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values#Summary-Plots)).

## testSHAP4AE_new.py
Plots the shap summary plots, considering the 5 variables that contribute the most to the reconstruction error in the event with the highest loss.

Note that it employs the get_layer method. In order for this method to access some particular layer of the encoder or decoder, the VAE model has to be defined as a whole: encoder and decoder can't be considered as separate layer objects (see /VAEmodel/VAE_new_model.py).

