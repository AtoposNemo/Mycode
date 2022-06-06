# -*- coding:utf-8 -*-

import numpy as np
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
import  pandas as pd
import  csv
import  gc
import itertools

class Encoding(object):
    def __init__(self):
        self.server_ip = "127.0.0.1"
        self.bert_client = BertClient(ip=self.server_ip)

    def encode(self, query):
        tensor = self.bert_client.encode([query])
        return tensor

    def query_similarity(self, query_list):
        tensors = self.bert_client.encode(query_list)
        return cosine_similarity(tensors)[0][1]
    
    
if __name__ == "__main__": 
    test = pd.read_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\最终数据\筛选后.csv')
    ec = Encoding()
    vectors=pd.DataFrame()
    print('开始处理！')
    for index, row in test.iterrows():
        
        vector=ec.encode(row['body'])       
        
        vectors=vectors.append(pd.DataFrame(vector))
        
        del vector,row
        gc.collect()
        
    test = test.drop(['body'],axis=1)
    vectors.index=range(0,len(vectors))
    test = pd.concat([test,vectors],axis=1,ignore_index=True)
    
    print('处理完成！')
    print(test.shape)
    test.to_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\最终数据\词向量数据.csv',index=False)

    