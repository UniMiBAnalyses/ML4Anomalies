import numpy as np
from matplotlib import pyplot as plt


loss5002D = np.loadtxt("loss_test_newModelDimenstions_MinMaxScaler_30_20_10_5_200.csv",delimiter=",")
loss500_log = np.loadtxt("loss_test_newModelDimenstions_MinMaxScaler_150_100_50_4_100.csv",delimiter=",")
loss1000_log_batch16 = np.loadtxt("loss_test_newModelDimenstions_MinMaxScaler_50_10_10_5_100.csv",delimiter=",")
loss1000_cut = np.loadtxt("loss_test_newModelDimenstions_MinMaxScaler_30_20_10_5_100.csv",delimiter=",")

epochs5002D = range(len(loss5002D))
epochs500 = range(len(loss500_log))
epochs1000 = range(len(loss1000_log_batch16))
epochs1000_cut = range(len(loss1000_cut))
ax = plt.figure(figsize=(10,5), dpi=100,facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.set_xlim(xmin =-10.,xmax=201)
ax.plot(epochs5002D,loss5002D,"r",linewidth=3,alpha=0.5,label="model 0")
ax.plot(epochs500,loss500_log,"g",linewidth=3,alpha=0.5, label = "model 1")
ax.plot(epochs1000,loss1000_log_batch16,"purple",linewidth=3,alpha=0.5, label = "Model 2")
ax.plot(epochs1000_cut,loss1000_cut,"orange",linewidth=3,alpha=0.5, label = "Model 3")
plt.ylabel("Training loss")
plt.xlabel("Epochs")

plt.legend()
plt.show()