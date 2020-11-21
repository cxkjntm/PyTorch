#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2020/11/14 11:35
# @Author  : lxy
import torch
import torch.nn as nn
class Representation(nn.Module):

    def __init__(self,hidden_size,category_num,input_size=256):
        super(Representation,self).__init__()
        self.rnn = nn.LSTM(
            input_size= 64,
            hidden_size= 64,
            num_layers=1,
            batch_first=True
        )
        self.gru = nn.GRU(
            input_size= 64,
            hidden_size= 64,
            num_layers=1,
            batch_first=True
        )
        self.qlj = nn.Linear(64, 64)
        self.softmax = nn.Softmax(dim=2)
        self.bn = nn.LayerNorm(normalized_shape=[1,input_size])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self,x):
        out1,_ = self.gru(x)
        # out1 = self.tanh(out1)
        # out2,_ = self.rnn(x)
        # out2 = self.tanh(out2)
        # x = self.tanh(x)
        # out = self.relu(out1)
        out = self.qlj(out1)
        out = self.softmax(out)
        return out


class Few_shot(nn.Module):

    def __init__(self):
        super(Few_shot,self).__init__()

        self.mlp =  nn.Sequential(
			nn.Linear(64, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 64),
			nn.ReLU(inplace=True)
		)

    def forward(self,x):
        x = self.mlp(x)
        return x

