# INDEX
#
# uploading losses and weights
# normalization
# plotting lf
# plotting lf with root
# plots (signal&bkg, significance)
# cW sensibility
# golden ratio


import tensorflow as tf

import ROOT
import sys
import numpy as np
import matplotlib
from array import array
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import auc
#from sklearn.metrics import roc_auc_score

#cW = 0.3
modelN = sys.argv[1]
DIM = sys.argv[2]
op = sys.argv[3]


def sigmaFunction(k, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
    nS = 0. #signal (lin + quad)
    nB = 0. #background
                      
    for i in range(len(lossSM)):
        if lossSM[i] > k:
            nB = nB + weightsSM[i]
    for i in range(len(lossLIN)):
        if lossLIN[i] > k:
            nS = nS + weightsLIN[i]*cW
    for i in range(len(lossQUAD)):
        if lossQUAD[i] > k:
            nS = nS + weightsQUAD[i]*cW*cW
    if nB == 0.:
        nB = 0.5
        nS = 0.
    nS = abs(nS)
    sigma = nS/np.sqrt(nB)
    return nS, nB, sigma


def sigmaComputation(start, stop, step, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
                        
    Sigma = []
    Cut = []
    Signal = []
    Bkg = []
    
    for k in np.arange(start,stop,step):
        k = round(k, 4)
        ns, nb, sig = sigmaFunction(k, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        
        if nb >= 1.: 
            Signal.append(ns)
            Bkg.append(nb)
            Sigma.append(sig)
            Cut.append(k)
        
    return Sigma, Cut, Signal, Bkg 
    
    
def sigma_max_err1(k, cW, lossSM, weightsSM, normSM, lossLIN, weightsLIN, normLIN, lossQUAD, weightsQUAD, normQUAD):
    kB = normSM
    kL = normLIN*cW
    kQ = normQUAD*cW*cW   
    # separate the proper weights from the normalization
    weightsSM = weightsSM/normSM
    weightsLIN = weightsLIN/normLIN
    weightsQUAD = weightsQUAD/normQUAD
    
    nL = 0.
    nQ = 0.
    nB = 0.
    
    for i in range(len(lossSM)):
        if lossSM[i] > k:
            nB = nB + weightsSM[i]
    for i in range(len(lossLIN)):
        if lossLIN[i] > k:
            nL = nL + weightsLIN[i]
    for i in range(len(lossQUAD)):
        if lossQUAD[i] > k:
            nQ = nQ + weightsQUAD[i]
    
    sigma = abs(nL*kL + nQ*kQ)/np.sqrt(nB*kB)
    errSigma = np.sqrt((1/abs(nB*kB))*(abs(nL)*pow(kL,2)+abs(nQ)*pow(kQ,2))*pow(np.sign(nL*kL+nQ*kQ),2)+pow((nL*kL+nQ*kQ),2)/(4*kB*pow(nB,2)))
    
    return nL, nQ, nB, sigma, errSigma

   

# if the maximum of the sigma(cW) is below 3, the corresponding cW value is considered not sensible to BSM data
   
def cWsensibility(start, stop, step, cWstart, cWstop, cWstep, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
    sensible = []
    notsensible = []
    
    cWarr = np.arange(cWstart, cWstop, cWstep)
    np.around(cWarr, 4)
    #print (cWarr)
    
    for i in range(len(cWarr)):
        sigma,_,_,_ = sigmaComputation(start, stop, step, cWarr[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        maxsigma = np.amax(sigma)
        if maxsigma < 3.:
            notsensible.append(cWarr[i])
        if maxsigma >= 3. :
            sensible.append(cWarr[i])
            #print ("per cW ", round(cWarr[i], 4), " abbiamo max pari a ", maxsigma)
    #print (len(notsensible)-1)
    #lastNS = notsensible[len(notsensible)-1]
    if len(notsensible) == 0 :
        lastNS = 0
    else:
        lastNS = notsensible[len(notsensible)-1]
    if len(sensible) == 0 :
        firstS = 0
    else:
        firstS = sensible[0]    
    
    return lastNS, firstS
    

def GoldenRatioSearch (a, b, err, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
    # returns the MAXIMUM VALUE of sigma(k)
    r = (np.sqrt(5)-1)/2
    a1 = b - r*(b-a)
    a2 = a + r*(b-a)
    
    while abs(b-a) > err :
        _,_,f1 = sigmaFunction(a1, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        _,_,f2 = sigmaFunction(a2, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        
        if f1 < f2 : # > for minimum
            a = a1
            a1 = a2
            f1 = f2
            a2 = a + r*(b-a)
        else :
            b = a2
            a2 = a1
            f2 = f1
            a1 = b - r*(b-a)
        
    x_opt = (a + b)/2
    sig,bkg,f_opt = sigmaFunction(x_opt, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    
    return x_opt, f_opt, sig, bkg 



############### uploading losses and weights ###############

lossSM = np.loadtxt("../sm"+str(modelN)+"_dim"+str(DIM)+"_lossSM_split.csv", delimiter=",") #test 
lossLIN = np.loadtxt("../"+str(op)+str(modelN)+"_dim"+str(DIM)+"_lossLIN_total.csv", delimiter=",")
lossQUAD = np.loadtxt("../"+str(op)+str(modelN)+"_dim"+str(DIM)+"_lossQUAD_total.csv", delimiter=",")

weightsSM = np.loadtxt("../sm"+str(modelN)+"_dim"+str(DIM)+"_weightsSM_split.csv", delimiter=",") #test
weightsLIN = np.loadtxt("../"+str(op)+str(modelN)+"_dim"+str(DIM)+"_weightsLIN_total.csv", delimiter=",")
weightsQUAD = np.loadtxt("../"+str(op)+str(modelN)+"_dim"+str(DIM)+"_weightsQUAD_total.csv", delimiter=",")


##################### normalization #########################

luminosity = 1000.*350. #luminosity expected in 1/pb

fSM = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
hSM = fSM.Get("SSWW_SM_nums")
xsecSM = hSM.GetBinContent(1)
sumwSM = hSM.GetBinContent(2)
normSM = 5.* xsecSM * luminosity / (sumwSM) # on test set (0.2*total)

fLIN = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuplesBSM/ntuple_SSWW_"+str(op)+"_LI.root")
hLIN = fLIN.Get("SSWW_"+str(op)+"_LI_nums")
xsecLIN = hLIN.GetBinContent(1)
sumwLIN = hLIN.GetBinContent(2)
normLIN = xsecLIN * luminosity / (sumwLIN)

fQUAD = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuplesBSM/ntuple_SSWW_"+str(op)+"_QU.root")
hQUAD = fQUAD.Get("SSWW_"+str(op)+"_QU_nums")
xsecQUAD = hQUAD.GetBinContent(1)
sumwQUAD = hQUAD.GetBinContent(2)
normQUAD = xsecQUAD * luminosity / (sumwQUAD)

print ("normSM", normSM)
print ("normLIN", normLIN)
print ("normQUAD", normQUAD)

weightsSM = weightsSM*normSM
weightsLIN = weightsLIN*normLIN # weights are not yet multiplied by cW
weightsQUAD = weightsQUAD*normQUAD # weights are not yet multiplied by cW*cW

###################### plotting lf #########################

lossBSM = np.concatenate((lossLIN, lossQUAD), axis=0)
lossmax = np.amax(lossBSM)
print  (lossmax)

######################################################
'''
wLIN3 = weightsLIN*0.3
wQUAD3 = weightsQUAD*0.3*0.3
weightsBSM3 = np.concatenate((wLIN3, wQUAD3), axis=0)
wLIN5 = weightsLIN*0.5
wQUAD5 = weightsQUAD*0.5*0.5
weightsBSM5 = np.concatenate((wLIN5, wQUAD5), axis=0)
wLIN1 = weightsLIN*0.1
wQUAD1 = weightsQUAD*0.1*0.1
weightsBSM1 = np.concatenate((wLIN1, wQUAD1), axis=0)
wLIN9 = weightsLIN*0.9
wQUAD9 = weightsQUAD*0.9*0.9
weightsBSM9 = np.concatenate((wLIN9, wQUAD9), axis=0)
lossALL = np.concatenate((lossSM, lossBSM), axis=0)
weightsALL1 = np.concatenate((weightsSM, weightsBSM1), axis=0)
weightsALL3 = np.concatenate((weightsSM, weightsBSM3), axis=0)
weightsALL5 = np.concatenate((weightsSM, weightsBSM5), axis=0)
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function: model "+str(modelN)+", dim "+str(DIM)+", operator "+str(op))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
#ax.hist(lossALL,bins=150,range=[0., 0.05], weights=weightsALL,histtype="step",color="blue",alpha=.6,linewidth=2,label ="ALL Loss cW=0.3")
ax.hist(lossSM,bins=150,range=[0., lossmax], weights=weightsSM,histtype="step",color="crimson",alpha=.6,linewidth=2,label ="SM test Loss")
#ax.hist(lossALL,bins=150,range=[0., 0.05],weights=weightsALL5,histtype="step",color="blue",alpha=.6,linewidth=2,label ="Loss SM+BSM cW=0.5")
#ax.hist(lossALL,bins=150,range=[0., 0.05],weights=weightsALL3,histtype="step",color="dodgerblue",alpha=.6,linewidth=2,label ="Loss SM+BSM cW=0.3")
#ax.hist(lossALL,bins=150,range=[0., 0.05],weights=weightsALL1,histtype="step",color="cyan",alpha=.6,linewidth=2,label ="Loss SM+BSM cW=0.1")
ax.hist(lossBSM,bins=150,range=[0., lossmax],weights=weightsBSM9,histtype="step",color="midnightblue",alpha=.6,linewidth=2,label ="BSM Loss "+str(op)+"=0.9")
ax.hist(lossBSM,bins=150,range=[0., lossmax],weights=weightsBSM5,histtype="step",color="blue",alpha=.6,linewidth=2,label ="BSM Loss "+str(op)+"=0.5")
ax.hist(lossBSM,bins=150,range=[0., lossmax],weights=weightsBSM3,histtype="step",color="dodgerblue",alpha=.6,linewidth=2,label ="BSM Loss "+str(op)+"=0.3")
ax.hist(lossBSM,bins=150,range=[0., lossmax],weights=weightsBSM1,histtype="step",color="cyan",alpha=.6,linewidth=2,label ="BSM Loss "+str(op)+"=0.1")
#ax.set_yscale("log")
plt.legend(loc=1)
ax.patch.set_facecolor("w")
#plt.savefig("./lossesFINAL_m"+str(modelN)+"_dim"+str(DIM)+"_"+str(op)+".png", bbox_inches='tight')
plt.close()
'''

################ plotting lfSM with root ####################
'''
h = ROOT.TH1D("h_sm", "h_sm", 150, 0., 0.05)
h.SetLineColor(ROOT.kRed)
h.FillN(len(loss), array('d', lossSM), array('d', weightsSM) )
h.Scale(norm)
ROOT.gStyle.SetOptStat(0)
leg = ROOT.TLegend(0.89, 0.89, 0.7, 0.7)
leg.SetBorderSize(0)
leg.AddEntry(h, "350fb^{-1}", "F")
c  = ROOT.TCanvas("c", "c", 1000, 1000)
h.Draw("hist")
c.SetLogy()
leg.Draw()
c.Draw()
print("\nRoot integral: ",h.Integral())
'''

######################### plots ################################
'''
cWarr = np.arange(0.1, 1., 0.1)
np.around(cWarr, 4)
for i in range((len(cWarr)-1)):
    sigma, cut, signal, bkg = sigmaComputation(0., round(lossmax, 2), 0.001, cWarr[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    #print ("\n",sigma)
    #print ("\n",cut)
    #print ("\n",signal)
    #print ("\n",bkg)
    
    sqrtbkg = np.sqrt(bkg)
    ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
    plt.suptitle("Signal and Bkg, model "+str(modelN)+", dim "+str(DIM)+", "+str(op)+" "+str(round(cWarr[i],2)))
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    ax.plot(cut, bkg, '-', linewidth = 1.5, color="blue", alpha=1., label="background")
    ax.plot(cut, sqrtbkg, '--', linewidth=1.5, color="cornflowerblue", alpha=1., label="sqrt(bkg)")
    ax.plot(cut, signal, '-', linewidth = 1.5, color="crimson", alpha=1., label="signal")
    #ax.set_yscale('log')
    ax.set_xlabel("cut on loss function")
    plt.legend()
    plt.savefig("./1signalandbkg"+str(modelN)+"_dim"+str(DIM)+"_"+str(op)+str(round(cWarr[i],2))+".png", bbox_inches='tight')
    plt.close()
    
    horizontal_line = np.array([3 for h in range(len(cut))])
    ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
    plt.suptitle("Significance, model "+str(modelN)+", dim "+str(DIM)+", "+str(op)+" "+str(round(cWarr[i],2)))
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    ax.plot(cut, sigma, '-', linewidth = 1.5, color="orange", alpha=1.)
    ax.set_xlabel("cut on loss function")
    ax.set_ylabel("S/sqrt(B)")
    plt.plot(cut,horizontal_line,"--",color="r")
    plt.savefig("./1significance"+str(modelN)+"_dim"+str(DIM)+"_"+str(op)+str(round(cWarr[i],2))+".png", bbox_inches='tight')
    plt.close()
'''
################## cW sensibility #############################
print ("computing cW sensibility")
# cWsensibility (start, stop, step, cWstart, cWstop, cWstep)
Lns1, Fs1 = cWsensibility(0., 0.04, 0.001, 0., 1., 0.1, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
#print ("loss of sensibility within ", Lns1, " - ", Fs1)
Lns2, Fs2 = cWsensibility(0., 0.04, 0.001, round(Lns1, 4), round(Fs1, 4), 0.01, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
#print ("loss of sensibility within ", Lns2, " - ", Fs2)
Lns3, Fs3 = cWsensibility(0., 0.04, 0.001, round(Lns2, 4), round(Fs2, 4), 0.001, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
print ("\nabs sigma, model ", modelN, "dim ", DIM)
print ("loss of sensibility within ", Lns3, " - ", Fs3)
#Lns4, Fs4 = cWsensibility(0., 0.4, 0.1, round(Lns3, 4), round(Fs3, 4), 0.0001, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
#print ("loss of sensibility within ", Lns4, " - ", Fs4)

# Fs3 is the first sensible value, 0.001 step

##################################### 

print ("\n\n ghe sem \n\n")
#cWcomp = np.arange(0.0, 1., 0.005)
cWcomp = np.arange(0.1, 1., 0.05)
#print (cWcomp)
cWcomp = np.around(cWcomp, decimals = 2)

maximum = []
cutMAX = []
sigMAX =[]
bkgMAX = [] 
nL1MAX =[]
nQ1MAX = []
nB1MAX = []
sigma1MAX = []
err1MAX = []


for i in range(len(cWcomp)):
    
    sigma,cut,_,_ = sigmaComputation(0., round(lossmax, 2), 0.001, cWcomp[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    #print ("\ncW ", cWcomp[i])
    cutmax, sigmamax, sig_max, bkg_max = GoldenRatioSearch(cut[0], cut[len(cut)-1], 0.00001, cWcomp[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    
    #print ("maximum sigma value is ", sigmamax, " +- nonperv", "  at cut = ", cutmax)
    maximum.append(sigmamax)
    cutMAX.append(cutmax)
    sigMAX.append(sig_max)
    bkgMAX.append(bkg_max)
    
    nL1, nQ1, nB1, sigma1, errSigma1 = sigma_max_err1(cutmax, cWcomp[i], lossSM, weightsSM, normSM, lossLIN, weightsLIN, normLIN, lossQUAD, weightsQUAD, normQUAD)
    
    nL1MAX.append(nL1)
    nQ1MAX.append(nQ1)
    nB1MAX.append(nB1)
    sigma1MAX.append(sigma1)
    err1MAX.append(errSigma1)
    
    
    #print ("finito")

print ("\ncW values are")
print (cWcomp)
print ("\nsigma, cut, signal, bkg")
print (maximum)
print (cutMAX)
print (sigMAX)
print (bkgMAX)

print ("\n\nsigma, error, lin, quad, bkg")
print (sigma1MAX)
print (err1MAX)
print (nL1MAX)
print (nQ1MAX)
print (nB1MAX)


print ("\n\nsigma max and error:")
for i in range(len(cutMAX)):
    print (sigma1MAX[i], " +- ", err1MAX[i])




'''
# dim 7
#loss of sensibility within  0.38  -  0.381
maximum1 = [0.2873443249848567, 0.5632386228478348, 0.9284590093152132, 1.3910916709279215, 1.9644572986791722, 2.6385196903446073, 3.412697495689491, 4.286056602842843, 5.264313253621542, 6.340542001560883, 7.517920495739998, 8.796072529104805, 10.17421703288262, 11.652579185194691, 13.231158986085191, 14.909956435558119, 16.688971533585597, 18.568204280169862]
# dim 5
#loss of sensibility within  0.35  -  0.351
maximum2 = [0.28708068337561243, 0.55306718368994, 0.9798664055643456, 1.529274236966499, 2.2032526812435496, 3.000429003829387, 3.9204531956644146, 4.963325256760934, 6.129045187097137, 7.417750571184993, 8.829319966997272, 10.363760768614199, 12.021072976018948, 13.801256589237042, 15.704311608232999, 17.730238033060818, 19.879035863680976, 22.150705100090484]
# dim 3
#loss of sensibility within  0.31  -  0.311
maximum3 = [0.3019896749595966, 0.6897169069501027, 1.2364047540124168, 1.9414837067835296, 2.804953765260629, 3.8268149294447054, 5.0070671993309706, 6.345710574934852, 7.842745056235392, 9.49817064324415, 11.31198733596546, 13.284195134391576, 15.414794038519167, 17.70378404836199, 20.151165163885864, 22.756937385154494, 25.521100712122777, 28.443655144783715]
# WO ETAJ1 AND ETAJ2:
#dim 7
#max1 =
#dim 5
#max2 =
#dim 3 
#max3 =
horizontal_line = np.array([3 for h in range(len(cWcomp))])
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("maximum sigma value as a function of "+str(op)+", model "+str(modelN)+", dim "+str(DIM))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(cWcomp, maximum, '-', linewidth = 1.5, color="blue", alpha=1., label="max")
#ax.plot(cWcomp, maximum1, '-', linewidth = 1.5, color="blue", alpha=1., label="dim 7")
#ax.plot(cWcomp, maximum2, '-', linewidth = 1.5, color="orange", alpha=1., label="dim 5")
#ax.plot(cWcomp, maximum3, '-', linewidth = 1.5, color="deepskyblue", alpha=1., label="dim 3")
plt.plot(cWcomp,horizontal_line,"--",color="r")
ax.set_xlabel(str(op))
ax.set_ylabel("sigma max")
plt.legend(loc=2)
plt.subplots_adjust(bottom=0.145)
#plt.yticks(np.arange(0., ., 2.))
plt.xticks(np.arange(0.1, 1., 0.05), rotation=50)
plt.savefig("./1maximumsigma"+str(modelN)+"_dim"+str(DIM)+"_"+str(op)+".png", bbox_inches='tight')
plt.close()
'''

   
print ("done")
#plt.show()
