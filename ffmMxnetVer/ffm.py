import pandas as pd
import numpy  as np
import os
import mxnet as mx
import mxnet.autograd as ag
import mxnet.ndarray as nd
np.set_printoptions(precision=2,linewidth=100)
from dataLoadTool import DataLoadTool,FFM,DataSets



if __name__ == "__main__":


    os.chdir("C:\\Users\\an\\Documents\\study\\ffmTutorial\\ffmInPython")
    # 准备资源中
    trainSet = pd.read_csv("train.tiny-Copy1.csv")
    trainSet = trainSet.dropna(how='any')
    trainSet = trainSet.loc[:, ['Label', 'I1', 'I2', 'C17', 'C18', 'C19']]
    trainSet.reset_index(inplace=True, drop=True)

    xCols = ['I1', 'I2', 'C17', 'C18', 'C19']
    categories = ['C17', 'C18', 'C19']
    continues = ['I1', 'I2']

    data = DataLoadTool( trainSet, 'Label', xCols ,categories , continues)

    ffm = FFM(data,testFrac = 0)
    ffm.fit(3)

