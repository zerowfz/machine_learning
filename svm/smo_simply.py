import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
def smoSimple(Xdata,Ydata,C,toler,maxIter):
    n,d = Xdata.shape
    alpha = np.zeros(n)
    b=0
    num_bin = np.ones(n)
    for iter in range(maxIter):
        #choose the worst alpha
        err  = np.dot(np.dot((alpha*Ydata),Xdata),Xdata.T)+b-Ydata #get the err for all sample
        kkt = ((alpha==0)&(Ydata*err>=0))|((alpha>0)&(alpha<C)&(abs(Ydata*err)<toler))|((alpha==C)&(Ydata*err<=0))
        err1 = err.copy()
        err1[kkt]=0
        ind = np.arange(n)
        err1 = err1**2
        i = np.argmax(err1)
        if not kkt.all():
            #random choose a j
            #pro = np.random.rand(1)
            #if pro<0.99:
            #    j = np.argmax(abs(err-err[i]))
            #else:
            j = np.random.randint(n)
            while i == j:
                j = np.random.randint(n)
            alpha_iold = alpha[i].copy()
            alpha_jold = alpha[j].copy()
            sample_i = Xdata[i,:]
            sample_j = Xdata[j,:]
            #the min 
            if Ydata[i] == Ydata[j]:
                high = min(C,alpha[i]+alpha[j])
                low = max(0,alpha[i]+alpha[j]-C)
            else :
                high = min(C,C+alpha[i]-alpha[j])
                low = max(0,alpha[i]-alpha[j])
            if high == low:
                continue
            alpha[i] -= Ydata[i]*(err[j]-err[i])/(2*np.dot(sample_i,sample_j)-\
                    np.dot(sample_i,sample_i)-\
                    np.dot(sample_j,sample_j))
            test_var = alpha[i].copy()
            #get the high ,low
            if alpha[i]<low:
                alpha[i] = low
            elif alpha[i]>high:
                alpha[i] = high
            alpha[j] = alpha_jold +  Ydata[i]*Ydata[j]*(alpha_iold-alpha[i])
            #理论上是不可能小于0的，但是有时候由于计算机误差有可能小于0
            if alpha[j]<0:
                alpha[j]=0
            assert alpha[i]>=0
            assert alpha[j]>=0
            support_ind = (alpha>0)&(alpha<C)
            b = (Ydata[support_ind]-np.dot(np.dot((alpha*Ydata).T,Xdata),\
            Xdata[support_ind,:].T)).sum()/support_ind.sum()
        else :
            break
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
alpha ,b = smoSimple(data,label,5,0.01,1000)
w = np.dot((alpha*label),data)
x = np.arange(2,8,0.1)
y = -(w[0]*x+b)/w[1]

support_vector = (alpha>0)*(alpha<0.6)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,y)
pos = (label>0)
neg = (label<0)
ax.scatter(data[pos,0],data[pos,1],marker='^')
ax.scatter(data[neg,0],data[neg,1],c='r')
for vector in data[support_vector,:]:
    circle = Circle(vector,0.5,facecolor='none',edgecolor=(0,0.8,0.8),linewidth=3,alpha=0.5)
    ax.add_patch(circle)
plt.show()
