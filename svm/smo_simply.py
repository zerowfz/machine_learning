import numpy as np
#import matplotlib.pyplot as plt
def smoSimple(Xdata,Ydata,C,toler,maxIter):
    n,d = Xdata.shape
    alpha = np.zeros(n)
    for iter in range(maxIter):
        #choose the worst alpha
        ind = np.argsort(np.abs(alpha-C/2))
        i = ind[-1]
        #random choose a j
        np.random.shuffle(ind[:-1])
        j = ind[0]
        alpha_iold = alpha[i].copy()
        alpha_jold = alpha[j].copy()
        sample_i = Xdata[i,:]
        sample_j = Xdata[j,:]
        # pred_i ,pred_j
        pred_i = np.dot(np.dot((alpha*Ydata).T,Xdata),sample_i)
        err_i = pred_i - Ydata[i]
        pred_j = np.dot(np.dot((alpha*Ydata).T,Xdata),sample_j)
        err_j = pred_j - Ydata[j]
        #the min 
        alpha[i] -= Ydata[i]*(err_j-err_i)/(2*np.dot(sample_i,sample_j)-\
                np.dot(sample_i,sample_i)-\
                np.dot(sample_j,sample_j))
        #get the high ,low
        if Ydata[i] == Ydata[j]:
            high = min(C,alpha[i]+alpha[j])
            low = max(0,alpha[i]+alpha[j]-C)
        else :
            high = min(C,C+alpha[i]-alpha[j])
            low = max(0,alpha[i]-alpha[j])
        if alpha[i]<low:
            alpha[i] = low
        elif alpha[i]>high:
            alpha[i] = high
        alpha[j] += Ydata[i]*Ydata[j]*(alpha_iold-alpha[i])
    # get the b
    support_ind = (alpha>0)*(alpha<C)
    b = (Ydata[support_ind]-np.dot(np.dot((alpha*Ydata).T,Xdata),\
            Xdata[support_ind,:].T)).sum()/support_ind.sum()
    return alpha ,b

def loadDataSet(filename):
    dataMat = [];labelMat = []
    fr =open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return np.array(dataMat),np.array(labelMat)

data,label = loadDataSet('testSet.txt')
alpha ,b = smoSimple(data,label,0.6,0.001,40)
'''
w = alpha*label,data 
plt.figure()
pos = (label>0)
neg = (label<0)
plt.plot(data[pos,0],data[pos,1],)
plt.plot(data[neg,0],data[neg,1],)
'''
