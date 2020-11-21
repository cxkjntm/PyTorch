#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2020/11/7 15:48
# @Author  : lxy
import torch
import torch.nn as nn
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel , out_channel , stride = 1 , downsample = None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channel= in_channel,out_channels= out_channel,
                               kernel_size= 3,stride= stride , padding= 1 , bias= False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLu()
        self.conv2 = nn.Conv2d(in_channel = out_channel, out_channel= out_channel,
                               kernel_size=3,stride= stride,padding= 1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        # 第一层3x3
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二层3x3
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expanson = 4

    def __init__(self,in_channel , out_channel , stride = 1, downsample = None):
        super(Bottleneck,self).__init__()
        self.relu = nn.ReLu()
        self.conv1 = nn.Conv2d(in_channels= in_channel, out_channels=out_channel,
                               kernel_size=1,stride=1,padding=0,bias= False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels= out_channel , out_channels= out_channel,
                               kernel_size= 3, stride = stride , padding=1 ,bias= False )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels= out_channel, out_channels=out_channel*self.expanson,
                               kernel_size= 1, stride = 1, bias= False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample


    def forward(self,x):
        identify = x
        if self.downsample is not None:
            identify = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identify
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    # @params:
    # block: BasicBlock or Bottleneck
    # blocks_num :
    # num_class : number of class need to be recognizeed

    def __init__(self,block,blocks_num,num_class= 1000, include_top = True):
        super(ResNet,self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.relu = nn.ReLu()

        self.conv1 = nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,
                               padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,blocks_num[0])
        self.layer2 = self._make_layer(block,128,blocks_num[1],stride = 2)
        self.layer3 = self._make_layer(block,256,blocks_num[2],stride = 2)
        self.layer4 = self._make_layer(block,512,blocks_num[3],stride = 2)
        if self.include_top:
            # avgpool
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            # Linear
            self.fc = nn.Linear(512 * block.expansion,num_class)

        for m in self.modules():
            if isinstance((m,nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight,mode = 'fan_out',nonlinearity='relu')


    # @param:
    # block : 18/34 -> BasicBlock or 50/101/152 -> Bottleneck
    # channel : rgb -> 3 or identify channel ,can be realized as depth
    # blocks_num : 18 -> [2,2,2,2] or 34 -> [3,4,6,3] or 50 -> [3,4,6,3] or
    #              101 -> [3,4,23,3] or 152 -> [3,8,36,3]
    # stride : 1 in inside or 2 in outside
    def _make_layer(self,block,channel,blocks_num,stride =1):
        # downsample
        downsample = None
        # due to in_channel differ to outchannel , we downsample ConvX_1 , X = 3,4,5
        if stride !=1 or self.in_channel != channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,channel*block.expansion,kernel_size=1,stride = stride,bias=False),
                nn.BatchNorm2d(channel* block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel,channel,downsample = downsample,stride = stride))
        self.in_channel = channel * block.expansion

        for _ in range(1,blocks_num):
            layers.append(block(self.in_channel,channel))

        return nn.Swquential(*layers)

    def forward(self, x):
        # conv1 : 7*7,64 ,stride=2
        # output : 112*112
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # conv1 : 3*3 stride=2 最大池化
        # output : 112*112
        x = self.maxpool(x)
        # conv2_x :
        # output : 56*56 (112+2-3)/2+1 =56
        x = self.layer1(x)
        # conv3_x :
        # output : 28*28
        x = self.layer2(x)
        # conv4_x :
        # output : 14*14
        x = self.layer3(x)
        # conv5_x:
        # output : 7*7
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x