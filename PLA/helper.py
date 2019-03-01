import numpy as np
import matplotlib.pyplot as plt

def Judge(X,y,w):
    """
    判别函数，判断所有数据是否分类完成
    """
    n = X.shape[0]
    #判断是否同号
    num = np.sum(X.dot(w) * y > 0)
    return num == n

def preprocess(data):
	"""
	数据预处理
	"""
	#获取维度
	n,d = data.shape
	#分离X
	X = data[:,:-1]
	#添加偏置项x0=1
	X = np.c_[np.ones(n),X]
	#分离y
	y = data[:,-1]

	return X,y

def PLA(X, y, eta=1, max_step=np.inf):
    """
    PLA算法，X，y为输入数据，eta为步长，默认为1，max_step为最多迭代次数，默认为无穷
    """
    #获取维度
    n,d = X.shape
    #初始化
    w = np.zeros(d)
    #记录迭代次数
    t = 0
    #记录元素下标
    i = 0
    #记录最后一个错误的下标
    last = 0
    while not(Judge(X, y, w)) and t < max_step:
        if np.sign(X[i, :].dot(w) * y[i]) <= 0:
            #迭代次数增加
            t += 1
            w += eta * y[i] * X[i,:]
            #更新最后一个错误
            last = i

        #移动到下一个元素
        i += 1
        #如果i达到n，重置为0
        if i == n:
        	i = 0

    return t,last,w


def f1(g, X, y, n, eta=1, max_step=np.inf):
    """
    运行g算法n次，统计平均迭代次数，eta为步长，默认为1，max_step为最多迭代次数，默认为无穷 
    """
    result = []
    data = np.c_[X, y]
    for i in range(n):
        np.random.shuffle(data)
        X = data[:, :-1]
        y = data[:, -1]
        result.append(g(X, y, eta=eta, max_step=max_step)[0])
    plt.hist(result, normed=True) 
    plt.xlabel("迭代次数") 
    plt.title("平均运行次数为"+str(np.mean(result))) 
    plt.show()





            

