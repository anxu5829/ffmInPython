import pandas as pd
import numpy as np
import re
import json
from operator import itemgetter
from collections import OrderedDict
from collections import Counter
import mxnet.ndarray as nd
import mxnet.autograd as ag
from scipy.sparse import csr_matrix,lil_matrix,triu,vstack
from  numpy.random import shuffle
import random



class DataLoadTool(object):
    def __init__(self,dataSet,targetName,xCols,
                    categories, continues,
                 ):
        self.dataSet = dataSet
        self.target  = self.dataSet[targetName]
        self.Xdata = self.dataSet.loc[:,xCols]
        self.Xdata.loc[:, categories] = self.Xdata.loc[:, categories].apply(lambda x: x.astype('category'))
        self.Xdata.loc[:, continues] = self.Xdata.loc[:, continues].apply(lambda x: x.astype('float32'))
        self.xdata_trnsf = pd.get_dummies(self.Xdata, prefix=categories, prefix_sep=":", sparse=True)
        self.label_v1 = [label.split(':')[0] for label in self.xdata_trnsf.columns.tolist()]
        self.label_v2 = self.xdata_trnsf.columns.tolist()
        self.index = pd.MultiIndex.from_arrays([self.label_v1, self.label_v2], names=['field_id', 'index_id'])
        self.xdata_trnsf.columns = self.index

        self.label_v1_dict = self.createIndex(self.label_v1)
        self.label_v2_dict = self.createIndex(self.label_v2)


        self.indexToFiled = dict(zip(itemgetter(*self.label_v2)(self.label_v2_dict),
                                itemgetter(*self.label_v1)(self.label_v1_dict)))

        self.dataSize     = self.Xdata.shape[0]

    def printDict(self,d):
        print(json.dumps(d, indent = 4))


    def createIndex(self,label):

        #print(label)
        orderCol = list(OrderedDict.fromkeys(label))
        #print(orderCol)
        label_dict = dict(zip(orderCol,range(len(orderCol))))
        return label_dict


class FFM(object):

    def __init__(self,data,learning_rate = 0.001,
                 lamda = 0.0033,epochs = 5,testFrac = 0.25,numOfK = 3
                 ):
        self.data = data
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.epochs = epochs
        # weightInit
        # tunning parameters
        self.numOfField = self.data.label_v1_dict.__len__()
        self.numOfIndex = self.data.label_v2_dict.__len__()
        self.numOfK = numOfK
        self.Wmatrix = nd.random_uniform(shape=(self.numOfK, self.numOfField, self.numOfIndex))
        self.Wmatrix.attach_grad()

        # generate data
        self.sample = list(range(self.data.dataSize))
        shuffle(self.sample)
        splt = int((1- testFrac)*self.data.dataSize)
        self.trainSet = DataSets(self.data , self.sample[:splt])
        self.testSet  = DataSets(self.data , self.sample[splt:])

    def getPhi(self,data,userNum):
        record = csr_matrix(data.loc[userNum,:].values)
        record = record.T
        record_matrix = record.dot(record.T)
        record_matrix = triu(record_matrix, format='lil')
        record_diag = range(record_matrix.shape[0])
        record_matrix[record_diag, record_diag] = 0
        record_matrix = record_matrix.tocoo()
        _data = record_matrix.data
        _row = record_matrix.row
        _col = record_matrix.col
        rowindex = itemgetter(*_row)(self.data.indexToFiled)
        colindex = itemgetter(*_col)(self.data.indexToFiled)
        phi = (self.Wmatrix[:, rowindex, _col] * self.Wmatrix[:, colindex, _row] * nd.array(_data)).sum()
        return phi

    def SGD(self):
        self.Wmatrix[:] = self.Wmatrix - self.learning_rate * self.Wmatrix.grad

    def fit(self,batch_size = 10):
        for e in range(self.epochs):
            total_loss = 0
            for data, label in self.trainSet.data_iter(batch_size):
                with ag.record():
                    batchloss = 0
                    labelCounter = 0
                    for userNum in data.index:
                        print(userNum)
                        phi = self.getPhi(data,userNum)
                        loss = nd.log(1 + nd.exp(-label.loc[userNum] * phi)) + self.lamda * (self.Wmatrix * self.Wmatrix).sum() / 2
                        batchloss = batchloss +  loss
                        labelCounter += 1
                batchloss.backward()
                self.Wmatrix[:] = self.Wmatrix - self.learning_rate * self.Wmatrix.grad / batch_size

    def getParams(self):
        return(self.Wmatrix,self.data.indexToFiled,self.data.label_v1_dict,self.data.label_v2_dict)




class DataSets(object):
    def __init__(self,data,sample):
        self.target = data.target[sample].copy()
        self.xdata_trnsf = data.xdata_trnsf.loc[sample,:].copy()
        self.userList = sample


    def data_iter(self,batch_size):
        # 产生一个随机索引
        num_examples = self.userList.__len__()
        idx = list(range(num_examples))
        random.shuffle(idx)
        for i in range(0, num_examples, batch_size):
            xdata = self.xdata_trnsf.iloc[idx[i:min(i+batch_size,num_examples)],:]
            y = self.target.iloc[idx[i:min(i+batch_size,num_examples)]]
            yield xdata,y


