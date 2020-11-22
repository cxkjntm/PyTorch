#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2020/11/14 11:35
# @Author  : lxy
import torch
import torch.nn as nn
class Representation(nn.Module):

    def __init__(self,category_num,in_features):
        super(Representation,self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1,out_channels=category_num,kernel_size=1)
        self.maxpooling = nn.MaxPool1d(kernel_size=5,stride=2,padding=2)
        self.bn1 = nn.BatchNorm1d(category_num)
        self.conv2 = nn.Conv1d(in_channels=category_num,out_channels=category_num,kernel_size=3,padding=1,stride=1)
        self.conv3 = nn.Conv1d(in_channels=category_num,out_channels=category_num,kernel_size=1)
        self.bn2 = nn.BatchNorm1d(category_num)
        self.convres = nn.Conv1d(in_channels=category_num*2,out_channels=category_num,kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=category_num,out_channels=1,kernel_size=1)
        self.bn3 = nn.BatchNorm1d(1)
        self.avgpooling = nn.AdaptiveAvgPool1d(16)
        self.linear = nn.Linear(16,category_num)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.conv1(x)
        x_back = x
        print("conv1: ",x.size())

        # conv1_
        x = self.conv2(x)
        print("conv2: ",x.size())
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        print("conv3:",x.size())
        # x = torch.cat((x,x_back),1)
        # x = self.convres(x)
        x = x + x_back
        x = self.relu(x)
        # conv1_ over

        # conv2_
        x = self.conv2(x)
        print("conv2: ", x.size())
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        print("conv3:", x.size())
        # x = torch.cat((x, x_back), 1)
        # x = self.convres(x)
        x = x + x_back
        x = self.relu(x)
        # conv2_ over

        # conv3_
        x = self.conv2(x)
        print("conv2: ", x.size())
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        print("conv3:", x.size())
        # x = torch.cat((x, x_back), 1)
        # x = self.convres(x)
        x = x+x_back
        x = self.relu(x)
        # conv3_ over


        x = self.conv4(x)
        x = self.bn3(x)
        print("conv4: ", x.size())
        x = self.avgpooling(x)
        print("avgpooling :",x.size())
        # print(x)
        # print(x.size())
        x = self.linear(x)
        print("linear: ",x.size())
        x = self.softmax(x)
        print("softmax: ",x.size())
        return x


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

