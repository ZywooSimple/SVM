# -*- coding=utf-8 -*-
import csv    #加载csv包便于读取csv文件
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from module1 import DataReader

def pre_process_svm(dataarray):
    print("数据预处理中")
    rand,rand_1,rand_11=[],[],[]
    label,label_1,label_11=[],[],[]
    k=0
    for i in dataarray:
        k=k+1
        for j in i:
            if j[2]<-1.169:
                label_11.append(0)
            else:
                label_11.append(1)
            
            j1=j.tolist()
            del j1[2]
            rand_11.append(j1)
#        if k>1000:
#            break
    rand=np.array(rand_11,dtype='float32')
    label=np.array(label_11,dtype=int)
    return rand,label

database = DataReader()
data_array, final_risk, test_array, test_risk = database.get_data_array()
rand_data,label_data=pre_process_svm(data_array)
rand_test,label_test=pre_process_svm(test_array)

svm = cv2.ml.SVM_create()
# ml 机器学习模块 SCM_create() 创建
svm.setType(cv2.ml.SVM_C_SVC) # svm type
svm.setKernel(cv2.ml.SVM_LINEAR) # line #线性分类器
svm.setC(0.01)
#训练SVM分类器
print("SVM训练中")
result= svm.train(rand_data,cv2.ml.ROW_SAMPLE,label_data)
#测试数据处理
#pt_data=np.vstack([rand_test[0],rand_test[1],rand_test[2]])
pt_data = np.array(rand_test,dtype='float32')
#print(pt_data)
par1,par2=svm.predict(pt_data)
print("________________")
i,correct,wrong,op=0,0,0,0
for kkk in label_test:
    op=op+1
for par in par2:
    if par==label_test[i]:
        correct=correct+1
    else:
        wrong=wrong+1
    i=i+1
print(i,op)
print("正确数： ",correct)
print("错误数： ",wrong)
cv2.waitKey(0)
