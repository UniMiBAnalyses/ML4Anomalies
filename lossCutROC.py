import numpy as np
from matplotlib import pyplot as plt


def effComputation(cW):
    lossSM = np.loadtxt("lossSM_"+str(cW)+".csv",delimiter=",")    
    lossBSM = np.loadtxt("lossBSM_"+str(cW)+".csv",delimiter=",")
    weightsSM= np.loadtxt("weight_SM_"+str(cW)+".csv",delimiter=",")
    weightsBSM= np.loadtxt("weight_BSM_"+str(cW)+".csv",delimiter=",")
    effSM = []
    effBSM = []
    for cut in np.arange(0.,0.05,0.00005):
        nSM = 0
        nBSM = 0
        for i in range(len(lossSM)):
            if lossSM[i] > cut:
                nSM = nSM+weightsSM[i]
                #nSM = nSM+1
        for i in range(len(lossBSM)):
            if lossBSM[i] > cut:                
                nBSM = nBSM+weightsBSM[i]
                #nBSM = nBSM+1

        effSM.append(1.*nSM/(weightsSM.sum()))
        effBSM.append(1.*nBSM/(weightsBSM.sum()))

    return lossSM,lossBSM,weightsSM,weightsBSM,effSM,effBSM

cW = "0.3"
oldNames = "vae_denselayers_withWeights_7D_latentDim_200epoch_batchsize16_log_eventFiltered_noWeights_"+str(cW)
print("model 0 ",oldNames)
lossSM,lossBSM,weightsSM,weightsBSM,effSM,effBSM = effComputation(oldNames)
newNames1= "vae_test_newModelDimenstions_MinMaxScaler_150_100_50_4_100_"+str(cW)
print("model 1 ",newNames1)
lossSM1,lossBSM1,weightsSM1,weightsBSM1,effSM1,effBSM1 = effComputation(newNames1)
newNames2= "vae_test_newModelDimenstions_MinMaxScaler_50_10_10_5_100_"+str(cW)
print("model 2 ",newNames2)
lossSM2,lossBSM2,weightsSM2,weightsBSM2,effSM2,effBSM2 = effComputation(newNames2)
newNames3= "vae_test_newModelDimenstions_MinMaxScaler_30_20_10_5_100_"+str(cW)
print("model 3 ",newNames3)
lossSM3,lossBSM3,weightsSM3,weightsBSM3,effSM3,effBSM3 = effComputation(newNames3)
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
#ax.set_ylim(ymax=10000)
ax.hist(lossBSM,bins=200,range=(0.,0.05),weights=weightsBSM,histtype="step",color="red",alpha=1.,linewidth=2,density =1,label ="BSM Loss 0")
ax.hist(lossSM,bins=200,range=(0.,0.05),weights=weightsSM,histtype="step",linestyle='dashed',color="red",alpha=1.,linewidth=2,density =1, label="SM Loss 0")
ax.hist(lossBSM1,bins=200,range=(0.,0.05),weights=weightsBSM1,histtype="step",color="green",alpha=1.,linewidth=2,density =1,label ="BSM Loss 1")
ax.hist(lossSM1,bins=200,range=(0.,0.05),weights=weightsSM1,histtype="step",linestyle='dashed',color="green",alpha=1.,linewidth=2,density =1, label="SM Loss 1")
ax.hist(lossBSM2,bins=200,range=(0.,0.05),weights=weightsBSM2,histtype="step",color="purple",alpha=1.,linewidth=2,density =1,label ="BSM Loss 2")
ax.hist(lossSM2,bins=200,range=(0.,0.05),weights=weightsSM2,histtype="step",linestyle='dashed',color="purple",alpha=1.,linewidth=2,density =1, label="SM Loss 2")
ax.hist(lossBSM3,bins=200,range=(0.,0.05),weights=weightsBSM3,histtype="step",color="orange",alpha=1.,linewidth=2,density =1,label ="BSM Loss 3")
ax.hist(lossSM3,bins=200,range=(0.,0.05),weights=weightsSM3,histtype="step",linestyle='dashed',color="orange",alpha=1.,linewidth=2,density =1, label="SM Loss 3")

ax.set_yscale('log')
plt.legend()


ax = plt.figure(figsize=(5,5), dpi=100,facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="minor")
ax.yaxis.grid(True, which="minor")
#ax.set_xlim(xmin =0.,xmax=1.05)
#ax.set_ylim(ymin =0.,ymax=1.05)
ax.scatter(effBSM,effSM,color = 'r',s=4,linewidths=0.5,alpha=0.5, label="model0 cW ="+cW)
ax.scatter(effBSM1,effSM1,color = 'g',s=4,linewidths=0.5,alpha=0.5, label="model1 cW ="+cW)
ax.scatter(effBSM2,effSM2,color = 'purple',s=4,linewidths=0.5,alpha=0.5, label="model2 cW ="+cW)
ax.scatter(effBSM3,effSM3,color = 'orange',s=4,linewidths=0.5,alpha=0.5, label="model3 cW ="+cW)
plt.xticks(np.arange(0,1.1,0.05))
plt.yticks(np.arange(0,1.05,0.05))
plt.plot([0.,1],[0.,1],color="r")
plt.xlabel("BSM Efficiency")
plt.ylabel("SM Efficiency")

plt.legend()
plt.show()