import numpy as np
#本函数用来实现朴素贝叶斯，用于垃圾文本的分类问题
def train_NBC(X,Y):
    num_sample = len(X)
    num_word = len(X[0])
    p_abusive = sum(Y)/num_sample
    train_1 = X[Y==1,:]
    train_0 = X[Y==0,:]
    p0 = (sum(train_0)+1)/(sum(sum(train_0))+2)
    p1 = (sum(train_1)+1)/(sum(sum(train_1))+2)
    return np.log(p0),np.log(p1),p_abusive

#本函数用来创建词汇表
def create_voc(dataset):
    voc = set([])
    for documents in dataset:
        voc = voc | set(documents)
    return list(voc)

#把输入的文档转换成词汇表示
def words2vec(voc,documents):
    vec = [0]*len(voc)
    for word in documents:
        if word in voc:
            vec[voc.index(word)]=1
        else:
            print('this word is not in the voc')
    return vec

def classify_NBC(document,p0,p1,p_abusive):
    p1 = sum(document*p1)+np.log(p_abusive)
    p0 = sum(document*p0)+np.log(1-p_abusive)
    if p1>p0:
        return 1
    else:
        return 0

if __name__=='__main__':
    trainX = [['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','i','love','him']]
    trainY =[0,1,0]
    voc = create_voc(trainX)
    print(voc)
    train_mat = []
    for i in trainX:
        train_mat.append(words2vec(voc,i))
    p0,p1,pc = train_NBC(np.array(train_mat),np.array(trainY))
    print(p0)
    print(p1)
    test_doc = ['my']
    test_vec1 = np.array(words2vec(voc,test_doc))
    print(classify_NBC(test_vec1,p0,p1,pc))


