#处理手写字体数据
import numpy as np
import pandas as pd
import glob
def get_data():
    #将文本数据变成所用的训练测试数据。
    #一个文本数据对应一行，文件名的第一个数字对应标签信息
    train_name = glob.glob(r'./trainingDigits/*.txt')
    test_name = glob.glob(r'./testDigits/*.txt')
    train_num = len(train_name)
    train_name.extend(test_name)
    all_num = len(train_name)
    label = np.zeros(all_num)
    data = np.zeros([all_num,1024])
    for n,i in enumerate(train_num):
        label[n] = i.split('/')[-1][0]
        out = pd.read_csv(i,header=None)
        data[:,n] = out.values.flatten()
    return data
    
        