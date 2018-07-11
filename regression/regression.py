#回归模型
#1.实现最简单的线性回归模型。
#局部线性加权回归模型。（无参模型）

import numpy as np
import pandas as pd


def simple_1(X,Y):
    #X: 输入数据
    #Y:输入标签
    xTx = np.dot(X.T,X)
    w = np.dot(np.dot(np.linalg.inv(xTx),X.T),Y)
    return w

def load_data(filename):
    data = pd.read_csv(filename,header=None)
    data = data.values
    return data[:,0:-1],data[:,-1]

def get_weight(test,X,kernel,opt):
    W = np.eye(X.shape[0])
    if kernel =='gauss':
        W = W* np.exp(-np.sum((test-X)**2,1)/(2*opt**2))
    return W

def lwlr(test,X,Y,k):
    #输入：
    #1.训练数据（X,Y)
    #2.测试样本
    W = get_weight(test,X,'gauss',k)
    #print(W)
    xTw = np.dot(X.T,W)
    xTwx = np.dot(xTw,X)
    w = np.dot(np.dot(np.linalg.inv(xTwx),xTw),Y)
    #print(np.dot(X,w))
    return np.dot(test,w)

def lwlr_test(test_array,X,Y,k=1):
    #test_array,test样本
    m = test_array.shape[0]
    Y_predict = np.zeros(m)
    for n,i in enumerate(test_array):
        #print(i)
        Y_predict[n] = lwlr(i,X,Y,k)
    return Y_predict

#岭回归：
def ridge(X,Y,lam):
    d = X.shape[1]
    xTx = np.dot(X.T,X)
    return np.dot(np.dot(np.linalg.inv(xTx + lam*np.eye(d)),X.T),Y)

def lasso_grad(X,Y,lam,opt,threshold):
    n,d = X.shape;
    #init w
    #w_pre = np.zeros(d)
    w_pre = np.random.rand(d)
    iter = 30
    for i in range(iter):
        tem = np.ones(d)
        tem[w_pre<0] = -1
        grad = -2*np.dot((Y-np.dot(X,w_pre)).T,X) + tem*lam
        w = w_pre - opt*grad
        if np.sum(abs(w-w_pre))<threshold:
            print('done')
            print(w)
            break
        #print(w)
        w_pre = w
    return w

def lasso_xy(X,Y,lam,threshold):
    n,d = X.shape
    w = np.random.rand(d)
    #w = np.ones(d)
    error_before = np.sum((Y-np.dot(X,w))**2);
    iter = 1000
    for i in range(iter):
        for j in range(d):
            tem = np.ones(d)
            tem[j]=0
            X_j = X[:,tem>0]
            w_j = w[tem>0]
            #if w[j]>=0:
            #    tem_lam = lam
            #else :
            #    tem_lam = -lam
            #print(tem_lam)
            tem_p = np.dot((Y - np.dot(X_j,w_j)).T,X[:,j]) 
            if tem_p < (-lam/2):
                w[j] = (tem_p + lam/2)/np.sum(X[:,j]**2)
            elif tem_p > lam/2:
                w[j] = (tem_p - lam/2)/np.sum(X[:,j]**2)
            else :
                w[j] = 0
            #w[j] = (np.dot((Y - np.dot(X_j,w_j)).T,X[:,j]) - tem_lam/2) / np.sum(X[:,j]**2)
            #print(w)
        error = np.sum((Y-np.dot(X,w))**2)
        if abs(error-error_before) < threshold:
            print(i)
            break
        error_before = error
        #print(np.dot(error.T,error) + np.sum(lam*np.abs(w)))
    return w

def forward_step(X,Y,step,iter_num):
    n,d = X.shape
    w = np.zeros(d)
    error = Y - np.dot(X,w)
    error_final = np.dot(error.T,error)
    w_out = np.zeros([iter_num,d])
    for i in range(iter_num):
        for j in range(d):
            for neg in [-1,1]:
                w[j] = w[j] + neg * step
                error = Y-np.dot(X,w)
                error_total = np.dot(error.T,error)
                if error_total < error_final:
                    error_final = error_total
                    print(w)
                else:
                    w[j] = w[j] - neg * step
        w_out[i] = w
    return w,w_out
                
def error(Y1,Y2):
    return np.mean((Y1-Y2)**2)
def test():
    X,Y = load_data('ex0.txt')
    w = simple_1(X,Y)
    y_predict = np.dot(X,w)
    y_predict2 = lwlr_test(X,X,Y,0.01)
    #print(np.corrcoef(y_predict,Y))
    print(w)

if __name__ == '__main__':
    test()

