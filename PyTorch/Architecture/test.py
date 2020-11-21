#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2020/11/10 17:47
# @Author  : lxy
import json
import Perprocess
import  os
import Eassy
import RNN
# 获取全局配置
Gobel_Configs = Perprocess.Load_Configs()
Data_Config = Gobel_Configs["Data_Config_test"]
Model_Config = Gobel_Configs["model_path"]

father_dir = os.listdir(Data_Config["Data_Dir"])
# 构造sentences
print(Data_Config)
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
essays = Eassy.Eassy(sentences,labels)
essayvector = essays.E2V()
# 加载超参
config_path = "../Hyperparameter.json"
# with open(config_path,encoding="utf8") as f:
#     configs = json.load(f)
#
# rnnnet = Architecture.Architecture(configs["Architecture"])
rnnnet = RNN.RNN([100,100,100,100])
run = RNN.Run_rnn(rnnnet,essayvector,essays.labels,rnnnet.init_Hidden())
# rnn run rnnnet 还没构建完成
