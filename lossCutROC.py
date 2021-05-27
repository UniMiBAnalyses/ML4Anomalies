import numpy as np
from matplotlib import pyplot as plt
cW = 0.3

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
        for i in range(len(lossBSM)):
            if lossBSM[i] > cut:
                nBSM = nBSM+weightsBSM[i]

        effSM.append(1.*nSM/(weightsSM.sum()))
        effBSM.append(1.*nBSM/(weightsBSM.sum()))

    return lossSM,lossBSM,weightsSM,weightsBSM,effSM,effBSM


lossSM,lossBSM,weightsSM,weightsBSM,effSM,effBSM = effComputation(cW)
lossSM_07,lossBSM_07,weightsSM_07,weightsBSM_07,effSM_07,effBSM_07 = effComputation(0.7)


ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
#ax.set_ylim(ymax=10000)
ax.hist(lossBSM,bins=100,range=(0.,0.05),weights=weightsBSM,histtype="step",color="red",alpha=1.,linewidth=2,density =1,label ="BSM Loss")
ax.hist(lossSM,bins=100,range=(0.,0.05),weights=weightsSM,histtype="step",color="blue",alpha=1.,linewidth=2,density =1, label="SM Loss")
#plt.hist(myloss_BSM2,bins=100,range=(0.,0.00015),histtype="step",color="green",alpha=1.)
plt.legend()


ax = plt.figure(figsize=(5,5), dpi=100,facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="minor")
ax.yaxis.grid(True, which="minor")
ax.set_xlim(xmin =0.,xmax=1.1)
ax.set_ylim(ymin =0.,ymax=1.1)
ax.scatter(effBSM,effSM,color = 'b')
ax.scatter(effBSM_07,effSM_07,color = 'g',s=1)
plt.xticks(np.arange(0,1.1,0.05))
plt.yticks(np.arange(0,1.1,0.05))
plt.plot([0.,1],[0.,1],color="r")
plt.xlabel("BSM Efficiency")
plt.ylabel("SM Efficiency")

#plt.legend()
plt.show()