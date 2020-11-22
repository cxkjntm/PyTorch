#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2020/11/10 17:55
# @Author  : lxy
import Perprocess
import os
from gensim.models import word2vec as w2v
import torch
from sklearn.model_selection import train_test_split

class Eassy:
    def __init__(self,sentences,labels):
        self.sentences = sentences
        self.labels = labels
        self.Ck = []
    def proccess(self):
        i = 0
        while(i<len(self.sentences)):

            if len(self.sentences[i]) >50:
                self.sentences[i]=self.sentences[i][5:45]
                i += 1
            else:
                # print(self.sentences[i])
                del self.sentences[i]
                del self.labels[i]
        print("sentence: ",self.sentences)


    def lenEassy(self):
        return len(self.labels)

    def split(self):
        return train_test_split(self.Ck,self.labels,test_size=0.3,random_state=0)

    def w2v(self):
        values = w2v.Word2Vec(self.sentences,min_count=1,size=64)
        values.save("Models/w2v.model")
        print("\n------------Calculating Completed!Saving Values Now!------------\n")
        return values


    def load_Vector(self):
        if os.path.exists("D:\\pycharm-projects\\PyTorch\\Models\\w2v.model"):
            return w2v.Word2Vec.load("Models/w2v.model")
        else:
            print("\n------------Not Found Values ! Calculating Values Now!------------\n")
            return self.w2v()

    def E2V(self):
        model = self.load_Vector()
        step = 0
        self.proccess()
        # print("="*80,"\n文字权重\n")
        for out in self.sentences:
            essay = []
            sum = torch.zeros(64,dtype=float)
            for inside in out:
                essay.append(model[inside])
                sum += model[inside]
            sum /= len(essay)*1.0
            self.Ck.append(sum.tolist())
            # print(self.Ck)
            step += 1
        # print("="*80)
        self.Ck = torch.tensor(self.Ck,dtype=torch.float)
        # print(self.Ck)

