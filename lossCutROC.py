import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


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
        #effSM.append(1.*nSM/(len(weightsSM)))
        #effBSM.append(1.*nBSM/(len(weightsBSM)))
   
    return lossSM,lossBSM,weightsSM,weightsBSM,effSM,effBSM

cW = "0.3"
#vae_test_newModelDimenstions_MinMaxScaler_150_100_50_7_100_
oldNames = "vae_test_newModelDimenstions_MinMaxScaler_20_10_7_3_100_noise_0.1_cW_"+str(cW)
legend0 = oldNames.replace("test_newModelDimenstions_MinMaxScaler_","")
print("model 0 ",oldNames)
lossSM,lossBSM,weightsSM,weightsBSM,effSM,effBSM = effComputation(oldNames)
#vae_test_newModelDimenstions_MinMaxScaler_50_10_10_5_100_
newNames1= "vae_test_newModelDimenstions_MinMaxScaler_30_20_10_5_100_noise_0.1_cW_"+str(cW)
legend1 = newNames1.replace("test_newModelDimenstions_MinMaxScaler_","")
print("model 1 ",newNames1)
lossSM1,lossBSM1,weightsSM1,weightsBSM1,effSM1,effBSM1 = effComputation(newNames1)

newNames2= "vae_test_newModelDimenstions_MinMaxScaler_150_100_50_4_200_"+str(cW)
legend2 = newNames2.replace("test_newModelDimenstions_MinMaxScaler_","")
print("model 2 ",newNames2)
lossSM2,lossBSM2,weightsSM2,weightsBSM2,effSM2,effBSM2 = effComputation(newNames2)

newNames3= "vae_test_newModelDimenstions_MinMaxScaler_30_20_10_5_100_"+str(cW)
legend3 = newNames3.replace("test_newModelDimenstions_MinMaxScaler_","")
print("model 3 ",newNames3)
lossSM3,lossBSM3,weightsSM3,weightsBSM3,effSM3,effBSM3 = effComputation(newNames3)

newNames4 = "vae_test_newModelDimenstions_MinMaxScaler_20_10_7_3_100_"+str(cW)
legend4 = newNames4.replace("test_newModelDimenstions_MinMaxScaler_","")
print("model 4 ",newNames4)
lossSM4,lossBSM4,weightsSM4,weightsBSM4,effSM4,effBSM4 = effComputation(newNames4)


numpy_dataSM3 = np.array(lossSM3)
numpy_dataSM4 = np.array(lossSM4)
dfLoss = pd.DataFrame(data=numpy_dataSM3, columns = ["SM3"])
dfLossSM4 = pd.DataFrame(data=numpy_dataSM4, columns = ["S4M"])
dfLoss['SM4'] =dfLossSM4
lossCorr = dfLoss.corr()
#import seaborn as sn
#sn.heatmap(lossCorr, annot=True)
#plt.show()

numpy_dataBSM3 = np.array(lossBSM3)
numpy_dataBSM4 = np.array(lossBSM4)
dfLossBSM = pd.DataFrame(data=numpy_dataBSM3, columns = ["BSM3"])
dfLossBSM4 = pd.DataFrame(data=numpy_dataBSM4, columns = ["BS4M"])
dfLossBSM['BSM4'] =dfLossBSM4
lossCorrBSM = dfLossBSM.corr()
#sn.heatmap(lossCorrBSM, annot=True)


ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
#ax.set_ylim(ymax=10000)
ax.hist(lossBSM,bins=200,range=(0.,0.05),weights=weightsBSM,histtype="step",color="red",alpha=1.,linewidth=2,density =1,label ="BSM "+legend0)
ax.hist(lossSM,bins=200,range=(0.,0.05),weights=weightsSM,histtype="step",linestyle='dashed',color="red",alpha=1.,linewidth=2,density =1, label="SM "+legend0)
ax.hist(lossBSM1,bins=200,range=(0.,0.05),weights=weightsBSM1,histtype="step",color="green",alpha=1.,linewidth=2,density =1,label ="BSM "+legend1)
ax.hist(lossSM1,bins=200,range=(0.,0.05),weights=weightsSM1,histtype="step",linestyle='dashed',color="green",alpha=1.,linewidth=2,density =1, label="SM "+legend1)
ax.hist(lossBSM2,bins=200,range=(0.,0.05),weights=weightsBSM2,histtype="step",color="purple",alpha=1.,linewidth=2,density =1,label ="BSM "+legend2)
ax.hist(lossSM2,bins=200,range=(0.,0.05),weights=weightsSM2,histtype="step",linestyle='dashed',color="purple",alpha=1.,linewidth=2,density =1, label="SM "+legend2)
ax.hist(lossBSM3,bins=200,range=(0.,0.05),weights=weightsBSM3,histtype="step",color="orange",alpha=1.,linewidth=2,density =1,label ="BSM "+legend3)
ax.hist(lossSM3,bins=200,range=(0.,0.05),weights=weightsSM3,histtype="step",linestyle='dashed',color="orange",alpha=1.,linewidth=2,density =1, label="SM "+legend3)
ax.hist(lossBSM4,bins=200,range=(0.,0.05),weights=weightsBSM4,histtype="step",color="blue",alpha=1.,linewidth=2,density =1,label ="BSM "+legend4)
ax.hist(lossSM4,bins=200,range=(0.,0.05),weights=weightsSM4,histtype="step",linestyle='dashed',color="blue",alpha=1.,linewidth=2,density =1, label="SM "+legend4)
ax.set_yscale('log')
plt.legend()
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
plt.style.use('seaborn-whitegrid')
plt.plot(lossBSM4,lossBSM3,'o', color='blue',alpha = 0.1)
plt.plot(lossSM4,lossSM3,'o', color='red',alpha = 0.1)
#plt.show()

ax = plt.figure(figsize=(7,5), dpi=100,facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="minor")
ax.yaxis.grid(True, which="minor")
#ax.set_xlim(xmin =0.,xmax=1.05)
#ax.set_ylim(ymin =0.,ymax=1.05)
ax.scatter(effBSM,effSM,color = 'r',s=4,linewidths=0.5,alpha=0.5, label=legend0)
ax.scatter(effBSM1,effSM1,color = 'g',s=4,linewidths=0.5,alpha=0.5, label=legend1)
ax.scatter(effBSM2,effSM2,color = 'purple',s=4,linewidths=0.5,alpha=0.5, label=legend2)
ax.scatter(effBSM3,effSM3,color = 'orange',s=4,linewidths=0.5,alpha=0.5, label=legend3)
ax.scatter(effBSM4,effSM4,color = 'blue',s=4,linewidths=0.5,alpha=0.5, label=legend4)
plt.xticks(np.arange(0,1.1,0.05))
plt.yticks(np.arange(0,1.05,0.05))
plt.plot([0.,1],[0.,1],color="black")
plt.xlabel("BSM Efficiency")
plt.ylabel("SM Efficiency")
legend = ax.legend(loc='best', shadow=True, fontsize='large', markerscale=3.)
#plt.legend()
plt.show()