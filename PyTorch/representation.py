#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2020/11/20 20:35
# @Author  : lxy
import numpy
import torch
import torch.utils.data as Data
import Perprocess
import  os
import Eassy
import Architecture
import matplotlib.pyplot as plt
# 获取全局配置
Gobel_Configs = Perprocess.Load_Configs()
Data_Config = Gobel_Configs["Few_shot_data"]
# Data_Config = Gobel_Configs["Data_Config"]
Few_shot_data = Gobel_Configs["Few_shot_data"]
father_dir = os.listdir(Data_Config["Data_Dir"])
def proc_data(x,y):
    essayTensor = x.clone().float()
    essayTensor = torch.unsqueeze(essayTensor, 1)
    labelsTensor = torch.tensor(y, dtype=torch.long)
    labelsTensor = torch.unsqueeze(labelsTensor, 1)
    labelsTensor = torch.unsqueeze(labelsTensor, 1)
    data = Data.TensorDataset(essayTensor, labelsTensor)
    loader = Data.DataLoader(
        dataset=data,
        batch_size=20,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    return loader

def load_data():
    # 构造sentences
    sentences = []
    labels =[]
    train_data = []
    label = 0
    for file  in father_dir:
        print(Data_Config["Data_Dir"],file)
        file_path = os.path.join(Data_Config["Data_Dir"] , file)
        with open(file_path,'r',encoding='utf-8') as f:
            data = [line.split() for line in f.readlines()]
        sentences.extend(data)
        labels.extend([label] * len(data))
        label += 1
    print("\n--------Building essays completed!----------\n")
    essays = Eassy.Eassy(sentences,labels)
    essays.E2V()
    # print("Ck: ",essays.Ck)
    print("\n--------Calculating essay vectors completed!----------\n")
    train_x,test_x,train_y,test_y = essays.split()
    dataLen_train = len(train_y)
    dataLen_test =len(test_y)
    train_loader = proc_data(train_x,train_y)
    test_loader = proc_data(test_x,test_y)
    return train_loader,dataLen_train,test_loader,dataLen_test



def Acc(prediciton,label):
    acc = float((prediciton == label).astype(int).sum())/float(label.size)
    return acc

def train(loader,iter):
    accList= [[],[]]
    for epoch in range(iter):
        for step,(batch_x, batch_y) in enumerate(loader):
            model.zero_grad()
            # print("batch_x",batch_x)
            out = model(batch_x)
            out = out.view(-1,out.shape[2])
            # print(out)
            pred = torch.max(out, 1)[1].data.numpy()
            print(pred)
            batch_y = batch_y.view(-1)
            loss = loss_fun(out,batch_y)
            label = numpy.asarray(batch_y,dtype=int,order=None)
            print(label)
            acc = Acc(pred,label)
            accList[step].append(acc)
            print("\n<<-----epoch:{0} batch:{1}  | batch_x_size:{2}  | loss:{3:.3f}  | acc:{4:.3f}----->>\n".format(epoch,step,batch_x.size(),loss,acc))
            loss.backward()
            opt.step()
    plt.plot(range(1,iter+1),accList[0],marker='*',mec='b',mfc='w')
    plt.plot(range(1,iter+1),accList[1],marker='s',mec='g',mfc='w')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_loader,dataLen_train,test_loader,dataLen_test = load_data()
    print("dataLen: ",dataLen_train)
    model = Architecture.Representation(5, dataLen_train)

    loss_fun = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    train(train_loader,300)
    for step, (batch_x, batch_y) in enumerate(test_loader):
        test = model.eval()
        out = test(test_loader)
        out = out.view(-1, out.shape[2])
        # print(out)
        pred = torch.max(out, 1)[1].data.numpy()
        print(pred)
        batch_y = batch_y.view(-1)
        loss = loss_fun(out, batch_y)
        label = numpy.asarray(batch_y, dtype=int, order=None)
        print(label)
        acc = Acc(pred, label)
        print("\n<<----- batch:{1} | loss:{3:.3f}  | acc:{4:.3f}----->>\n".format( step,loss, acc))