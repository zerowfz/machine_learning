'''
此处加入了kernel,加入kernel实际上即是对x*x.T的求解变成了kernel(x*x.T)
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import glob

def trans(a,b,kernel):
    n1 = a.shape[0]
    n2 = b.shape[0]
    if b.size == b.shape[0]:
        n2 = 1
    if kernel[0] == 'lin':
        return np.dot(a,b.T)
    elif kernel[0] == 'gaussian':
        if n2==1:
            tem = np.zeros(n1)
            tem += ((a-b)**2).sum(1)
        else:
            tem = np.zeros((n1,n2))
            for i in range(n2):
                tem[:,i] += ((a-b[i,:])**2).sum(1)
        return np.exp(-tem/kernel[1])
    else:
        print("error")

class optStruct:
    def __init__(self,Xdata,Ydata,C,toler,maxIter,kernel):
        self.X = Xdata
        self.Y = Ydata
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.n = Xdata.shape[0]
        self.kernel = kernel
        self.K = self.ktrans()

    def ktrans(self):
        #kernel [0]:表示kernel名字
        #后面的表示kernel的参数
        if self.kernel[0] == 'lin':
            return np.dot(self.X,self.X.T)
        elif self.kernel[0] == 'gaussian':
            tem = np.zeros((self.n,self.n))
            for i in range(self.n):
                tem[:,i] += ((self.X-self.X[i,:])**2).sum(1)
            return np.exp(-tem/self.kernel[1])
        else :
            print("error")
        

def smoSimple(Xdata,Ydata,C,toler,maxIter,kernel):
    os = optStruct(Xdata,Ydata,C,toler,maxIter,kernel)
    n,d = Xdata.shape
    alpha = np.zeros(n)
    b=0
    for iter in range(maxIter):
        #choose the worst alpha
        err  = np.dot(alpha*Ydata,os.K)+b-Ydata #get the err for all sample
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
            alpha[i] -= Ydata[i]*(err[j]-err[i])/(2*os.K[i,j]-os.K[i,i]-os.K[j,j])
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
            b = (Ydata[support_ind]-np.dot(alpha*Ydata,os.K[:,support_ind])).sum()/support_ind.sum()
        else :
            break
    return os,alpha ,b

def loadDataSet(filename):
    dataMat = [];labelMat = []
    fr =open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return np.array(dataMat),np.array(labelMat)

def getDigitSet(trainpath,testpath):
    trainfile = glob.glob(trainpath+'*.txt')
    testfile = glob.glob(testpath+'*.txt')
    traindata = []
    trainY = []
    testdata = []
    testY=[]
    for i in trainfile:
        tem=[]
        if i.split('/')[-1][0]=='1':
            trainY.append(1)
        else:
            trainY.append(-1)
        with open(i,'r') as f:
            for j in f.readlines():
                tem.extend(j.strip())
        traindata.append(tem)
    for i in testfile:
        tem=[]
        if i.split('/')[-1][0]=='1':
            testY.append(1)
        else:
            testY.append(-1)
        with open(i,'r') as f:
            for j in f.readlines():
                tem.extend(j.strip())
        testdata.append(tem)
    return np.array(traindata,np.int),np.array(trainY,np.int),np.array(testdata,np.int),\
            np.array(testY,np.int)
        
def testfordigit(trainpath,testpath,kernel=('gaussian',100)):
    X,Y,X1,Y1 = getDigitSet(trainpath,testpath)
    #print(X,Y,X1,Y1)
    os,alpha,b = smoSimple(X,Y,200,0.0001,10000,kernel)
    supind = alpha>0
    #print(alpha)
    supve = X[supind,:]
    print("support vector:",supve.shape[0])
    K = trans(supve,X1,kernel)
    c=alpha[supind]
    d = Y[supind]
    predict = np.dot(c*d,K)+b
    print((np.sign(predict)==Y1).sum()/X1.shape[0])

testfordigit('./trainingDigits/','./testDigits/',kernel=['lin'])
testfordigit('./trainingDigits/','./testDigits/',kernel=['gaussian',0.1])
testfordigit('./trainingDigits/','./testDigits/',kernel=['gaussian',1])
testfordigit('./trainingDigits/','./testDigits/',kernel=['gaussian',10])
testfordigit('./trainingDigits/','./testDigits/',kernel=['gaussian',100])
testfordigit('./trainingDigits/','./testDigits/',kernel=['gaussian',1000])
'''
if __name__='__main__':
    data,label = loadDataSet('testSetRBF.txt')
    os,alpha ,b = smoSimple(data,label,200,0.0001,10000,('gaussian',1.3))

    #support_vector = (alpha>0)*(alpha<0.6)
    predict = np.dot(alpha*label,os.K)+b
    print(predict)
    print(label)
    print((np.sign(predict)==label).sum()/data.shape[0])

    support_vector = (alpha>0)&(alpha<200)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.plot(x,y)
    pos = (label>0)
    neg = (label<0)
    ax.scatter(data[pos,0],data[pos,1],marker='^')
    ax.scatter(data[neg,0],data[neg,1],c='r')
    for vector in data[support_vector,:]:
        circle = Circle(vector,0.01,facecolor='none',edgecolor=(0,0.8,0.8),linewidth=3,alpha=0.5)
        ax.add_patch(circle)
    plt.show()
'''
