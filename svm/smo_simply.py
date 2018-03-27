import numpy as np
from numpy import mat
def smoSimple(Xdata,Ydata,C,toler,maxIter):
    Xdata = np.array(Xdata)
    n,d = Xdata.shape
    Ydata = np.array(Ydata).reshape(n,1)
    alpha = np.zeros((n,1))
    for iter in range(maxIter):
        #choose the worst alpha
        ind = np.argsort(np.abs(alpha[:,0]-C/2))
        i = ind[-1]
        #random choose a j
        np.random.shuffle(ind[:-1])
        j = ind[0]
        alpha_iold = alpha[i][0].copy()
        alpha_jold = alpha[j][0].copy()
        sample_i = Xdata[i,:].reshape(d,1)
        sample_j = Xdata[j,:].reshape(d,1)
        # pred_i ,pred_j
        pred_i = np.dot(np.dot((alpha*Ydata).T,Xdata),sample_i)
        err_i = pred_i - Ydata[i][0]
        pred_j = np.dot(np.dot((alpha*Ydata).T,Xdata),sample_j)
        err_j = pred_j - Ydata[j][0]
        #the min 
        assert 1==0
        alpha[i] -= Ydata[i][0]*(err_j-err_i)/(2*np.dot(sample_i.T,sample_j)-\
                np.dot(sample_i.T,sample_i)-\
                np.dot(sample_j.T,sample_j))
        #get the high ,low
        if Ydata[i][0] == Ydata[j][0]:
            high = np.min(C,alpha[i]+alpha[j])
            low = np.max(0,alpha[i]+alpha[j]-C)
        else :
            high = np.min(C,C+alpha[i]-alpha[j])
            low = np.max(0,alpha[i]-alpha[j])
        if alpha[i]<low:
            alpha[i] = low
        elif alpha[i]>high:
            alpha[i] = high
        alpha[j] += Ydata[i][0]*Ydata[j][0]*(alpha_iold-alpha[i])
    # get the b
    support_ind = (((alpha>0)+(alpha<C))-1)>0
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
    return dataMat,labelMat

data,label = loadDataSet('testSet.txt')
alpha ,b = smoSimple(data,label,0.6,0.001,40)

