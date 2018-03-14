import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#X:n*d , Y:n*1
def logistic(X,Y):
    array_X = np.c_[X,np.ones([X.shape[0],1])]
    n,d = array_X.shape
    alpha = 0.001
    maxiter = 1000
    weights = np.ones((d,1))
    test_w1 = []
    test_w2 = []
    test_w3 = []
    for i in range(maxiter):
        #alpha = 4/(i+1)+1
        pred = sigmoid(np.dot(array_X,weights))
        grad = np.dot(array_X.T,(Y-pred))
        weights = weights+alpha*grad
        test_w1.append(weights[0,0])
        test_w2.append(weights[1,0])
        test_w3.append(weights[2,0])
        print(weights)
    fig = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(np.arange(len(test_w1)),test_w1)
    plt.subplot(3,1,2)   
    plt.plot(np.arange(len(test_w2)),test_w2)
    plt.subplot(3,1,3)
    plt.plot(np.arange(len(test_w3)),test_w3)
    plt.xlabel('alpha = :'+str(alpha))
    plt.show()
    return weights

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def plot_data(X,Y,weight):
    c_1 = (Y==1)[:,0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[c_1,0],X[c_1,1],s=30,c='red',marker='s')
    ax.scatter(X[~c_1,0],X[~c_1,1],s=30,c='green')
    x = np.arange(-3,3,0.1)
    y = (-weight[0]*x-weight[2])/weight[1]
    ax.plot(x,y)
    plt.show()
    
def stocGrad(X,Y,maxIter,batch_num):
    array_X = np.c_[X,np.ones([X.shape[0],1])]
    n,d = array_X.shape
    s = int(n/batch_num)
    row = np.arange(n)
    alpha = 0.001
    weights = np.ones((d,1))
    test_w1 = []
    test_w2 = []
    test_w3 = []
    for j in range(maxIter):
        #np.random.shuffle(row)
        for i in range(s):
            #alpha = 4/(1.0+j+i)+0.01
            train_x = array_X[row[i*batch_num:(i+1)*batch_num],:]
            train_y = Y[row[i*batch_num:(i+1)*batch_num],:]
            pred = sigmoid(np.dot(train_x,weights))
            #print(train_x)
            #print(train_y)
            #assert 1==0
            grad = np.dot(train_x.T,(train_y-pred))
            weights = weights+alpha*grad
            print(weights)
            test_w1.append(weights[0,0])
            test_w2.append(weights[1,0])
            test_w3.append(weights[2,0])
            #print(weights)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(np.arange(len(test_w1)),test_w1)
    plt.subplot(3,1,2)   
    plt.plot(np.arange(len(test_w2)),test_w2)
    plt.subplot(3,1,3)
    plt.plot(np.arange(len(test_w3)),test_w3)
    plt.show()
    return weights




X=[]
Y=[]
data = pd.read_csv('testSet.txt',delimiter='\t').values
X = data[:,:-1]
Y = data[:,-1]
Y = Y.reshape(Y.size,1)
weight= logistic(X,Y)
#weight2 = stocGrad(X,Y,500,1)
#weight3 = stocGrad(X,Y,500,3)
plot_data(X,Y,weight)
#plot_data(X,Y,weight2)
#plot_data(X,Y,weight3)
