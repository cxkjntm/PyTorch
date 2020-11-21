#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2020/11/10 17:47
# @Author  : lxy
import json

import numpy
import torch

import Perprocess
import  os
import Eassy
import Architecture
import matplotlib.pyplot as plt
# 获取全局配置
Gobel_Configs = Perprocess.Load_Configs()
Data_Config = Gobel_Configs["Data_Config"]
# Model_Config = Gobel_Configs["model_path"]
Few_shot_data = Gobel_Configs["Few_shot_data"]
father_dir = os.listdir(Data_Config["Data_Dir"])
# 构造sentences
sentences = []
labels =[]
label = 0
for file  in father_dir:
    file_path = os.path.join(Data_Config["Data_Dir"] , file)
    try:
        with open(file_path,'r',encoding='utf-8') as f:
            data = [line.split() for line in f.readlines()]

    except:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [line.split() for line in f.readlines()]
    sentences.extend(data)
    labels.extend([label]*len(data))
    label += 1
print("\n--------Building essays completed!----------\n")
essays = Eassy.Eassy(sentences,labels)
essays.E2V()
print("\n--------Calculating essay vectors completed!----------\n")
train_x,test_x,train_y,test_y = essays.split()

rnn = Architecture.Representation(64,64,64)


# loss function
loss_fun = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(rnn.parameters(),lr=0.01)
print("\n--------Starting train!----------\n")

acclist = []
for epoch in range(200):
    lab = []
    pred =[]
    for essay,label in zip(train_x,train_y):
        essay = torch.unsqueeze(essay, 0)
        essay = torch.unsqueeze(essay, 0)
        essay = essay.float()
        label = torch.tensor([label])
        lab.append(label.item())
        output = rnn(essay)
        output = output.squeeze(0)
        loss = loss_fun(output,label)
        loss.item()
        pred_y = torch.max(output, 1)[1].data.numpy()
        pred.append(pred_y.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    prednp = numpy.asarray(pred, dtype = int, order = None)
    labnp = numpy.asarray(lab, dtype = int, order = None)
    acc = float((prednp == labnp).astype(int).sum())/float(labnp.size)
    acclist.append(acc)
    print('Epoch: ',(epoch),"train loss:%.3f"%loss.data.numpy(),"acc:%.3f"%(acc),"\n")
    pred = []
    lab = []


lab = []
pred =[]

for test,label in zip(test_x,test_y):
    lab.append(label)
    rnn_test = rnn.eval()
    test = torch.unsqueeze(test,0)
    test = torch.unsqueeze(test,0)
    test = test.float()
    pred_y = rnn_test(test)
    pred_y = pred_y.squeeze(0)
    pred_y = torch.max(pred_y, 1)[1].data.numpy()
    pred.append(pred_y.item())
    # print("pred_y: ",pred_y," :: label : ",label,"\n")
prednp = numpy.asarray(pred, dtype = int, order = None)
labnp = numpy.asarray(lab, dtype = int, order = None)
acc = float((prednp == labnp).astype(int).sum())/float(labnp.size)
print("test acc:%.3f"%(acc),"\n")

plt.plot(range(200),acclist,marker='o',mec='r',mfc='w')
plt.xlabel("acc")
plt.ylabel("epoch")
plt.show()



