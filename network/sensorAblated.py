from typing import Union, NamedTuple
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from clstm import *
from cgru import *

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class TimeDistributedDown(nn.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first
    

    def forward(self, images):
         # Squash samples and timesteps into a single axis
        batch, timesteps, channels, height, width = images.size()
        
        x_reshape = images.contiguous().view(batch*timesteps, channels, height, width)  # (batch*timestep, channels, height, width)

        drop5, drop4, x1, x2, x3 = self.module(x_reshape)
        _, channels, height, width = drop5.size()
        # We have to reshape Y
        if self.batch_first:
            drop5 = drop5.contiguous().view(batch, timesteps, channels, height, width)  # (samples, timesteps, output_size)
        else:
            drop5 = drop5.view(timesteps, batch, channels, height, width)  # (timesteps, samples, output_size)

        return drop5, drop4, x1, x2, x3

class TimeDistributedUp(nn.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first
    

    def forward(self, images, drop4, x1, x2, x3):
         # Squash samples and timesteps into a single axis
        batch, timesteps, channels, height, width = images.size()
        
        x_reshape = images.contiguous().view(batch*timesteps, channels, height, width)  # (batch*timestep, channels, height, width)

        y = self.module(x_reshape, drop4, x1, x2, x3)
        _, channels, height, width = y.size()
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(batch, timesteps, channels, height, width)  # (samples, timesteps, output_size)
        else:
            y = y.view(timesteps, batch, channels, height, width)  # (timesteps, samples, output_size)

        return y
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
            self.conv1 = nn.Conv2d(in_channels, in_channels//2, 2)
            self.conv = DoubleConv(out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.relu(self.conv1(x1))
        # print(x1.size())
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.ZeroPad2d((diffX // 2,diffX - diffX // 2,diffY // 2,diffY - diffY // 2))(x1)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #x = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.conv2(x)

class unetDown(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        #factor = 1 if bilinear else 1
        self.down4 = Down(512, 1024)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        drop4 = nn.Dropout()(x4)
        x5 = self.down4(drop4)
        drop5 = nn.Dropout()(x5)

        return drop5, drop4, x1, x2, x3

class unetUp(nn.Module):
    def __init__(self, n_classes ,bilinear=True):
        super().__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
      #  self.p = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
    
    def forward(self, images, drop4, x1, x2, x3):
        x = self.up1(images, drop4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
     #   x = F.relu(self.p(x))

        return x


class SensorAblated(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()

        self.unetDown = unetDown(n_channels)
        self.unetUp = unetUp(n_classes, bilinear)
        
        self.tdDown = TimeDistributedDown(self.unetDown)
       # self.blstm1 = ConvBLSTM(in_channels=512, hidden_channels=1024, kernel_size=(3, 3), batch_first=True)
        self.tdUp = TimeDistributedUp(self.unetUp)
        # self.blstm2 = ConvBLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)
        # self.bcgru = ConvBGRU(in_channels=64, hidden_channels=64, kernel_size=(3, 3), num_layers=1 ,batch_first=True)
        #self.clstm = ConvLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3),batch_first=True)
        #self.cgru = ConvGRU(in_channels=64, hidden_channels=64,kernel_size=3, num_layers=1,batch_first=True)
        self.bcgru = ConvBGRU(in_channels=64, hidden_channels=64, kernel_size=3, num_layers=1 ,batch_first=True)

        self.outc = OutConv(64, 1)

    def forward(self, x):

        drop5, drop4, x1, x2, x3 = self.tdDown(x)

       # print(f'the outsize is {drop4.size()}')

        # rev_index = list(reversed([i for i in range(drop4.size(1))]))
        # reversed_out = drop4[:,rev_index,...]

        # #print(f'rev size: {reversed_out.size()}')

        # out = self.blstm1(drop4, reversed_out)

        # batch, timesteps, channels, height, width = drop5.size()
        # drop5 = drop5.contiguous().view(batch*timesteps, channels, height, width) 

        out = self.tdUp(drop5, drop4, x1, x2, x3)

        # rev_index = list(reversed([i for i in range(out.size(1))]))
        # reversed_out = out[:,rev_index,...]

        # out = self.blstm2(out, reversed_out)

        # # out = torch.sum(out, dim=1)
        reversed_idx = list(reversed(range(out.shape[1])))
        # rev_index = list(reversed([i for i in range(out.size(1))]))
        # reversed_out = out[:,rev_index,...]

        # out = self.blstm2(out, reversed_out)
        out_rev = out[:,reversed_idx,...]
        # # out = torch.sum(out, dim=1)
        out = self.bcgru(out, out_rev)

        #final_out = out[:,-1,...]

        logits = self.outc(out[:,-1,...])


        return logits

class SensorAblatedLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()

        self.unetDown = unetDown(n_channels)
        self.unetUp = unetUp(n_classes, bilinear)
        
        self.tdDown = TimeDistributedDown(self.unetDown)
       # self.blstm1 = ConvBLSTM(in_channels=512, hidden_channels=1024, kernel_size=(3, 3), batch_first=True)
        self.tdUp = TimeDistributedUp(self.unetUp)
        # self.blstm2 = ConvBLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)
        self.bcgru = ConvBGRU(in_channels=64, hidden_channels=64, kernel_size=(3, 3), num_layers=1 ,batch_first=True)
        #self.clstm = ConvLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3),batch_first=True)
        #self.clstm = ConvLSTM(in_channels=64, hidden_channels=64,kernel_size=(3,3),batch_first=True)

        self.outc = OutConv(64, 1)

    def forward(self, x):

        drop5, drop4, x1, x2, x3 = self.tdDown(x)

       # print(f'the outsize is {drop4.size()}')

        # rev_index = list(reversed([i for i in range(drop4.size(1))]))
        # reversed_out = drop4[:,rev_index,...]

        # #print(f'rev size: {reversed_out.size()}')

        # out = self.blstm1(drop4, reversed_out)

        # batch, timesteps, channels, height, width = drop5.size()
        # drop5 = drop5.contiguous().view(batch*timesteps, channels, height, width) 

        out = self.tdUp(drop5, drop4, x1, x2, x3)
        reversed_idx = list(reversed(range(out.shape[1])))
        # rev_index = list(reversed([i for i in range(out.size(1))]))
        # reversed_out = out[:,rev_index,...]

        # out = self.blstm2(out, reversed_out)
        out_rev = out[:,reversed_idx,...]
        # # out = torch.sum(out, dim=1)
        out = self.bcgru(out, out_rev)

        #final_out = out[:,-1,...]

        logits = self.outc(out[:,-1,...])


        return logits

class SensorAblatedTest(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()

        self.unetDown = unetDown(n_channels)
        self.unetUp = unetUp(n_classes, bilinear)
        
        self.tdDown = TimeDistributedDown(self.unetDown)
       # self.blstm1 = ConvBLSTM(in_channels=512, hidden_channels=1024, kernel_size=(3, 3), batch_first=True)
        self.tdUp = TimeDistributedUp(self.unetUp)
        # self.blstm2 = ConvBLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)
        # self.bcgru = ConvBGRU(in_channels=64, hidden_channels=64, kernel_size=(3, 3), num_layers=1 ,batch_first=True)
        #self.clstm = ConvLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3),batch_first=True)
        self.bcgru = ConvBGRU(in_channels=64, hidden_channels=64, kernel_size=3, num_layers=1 ,batch_first=True)
        #self.cgru = ConvGRU(in_channels=64, hidden_channels=64,kernel_size=3, num_layers=1,batch_first=True)

        self.outc = OutConv(64, 1)

    def forward(self, x):

        drop5, drop4, x1, x2, x3 = self.tdDown(x)

       # print(f'the outsize is {drop4.size()}')

        # rev_index = list(reversed([i for i in range(drop4.size(1))]))
        # reversed_out = drop4[:,rev_index,...]

        # #print(f'rev size: {reversed_out.size()}')

        # out = self.blstm1(drop4, reversed_out)

        # batch, timesteps, channels, height, width = drop5.size()
        # drop5 = drop5.contiguous().view(batch*timesteps, channels, height, width) 

        out = self.tdUp(drop5, drop4, x1, x2, x3)

        # rev_index = list(reversed([i for i in range(out.size(1))]))
        # reversed_out = out[:,rev_index,...]

        # out = self.blstm2(out, reversed_out)

        # # out = torch.sum(out, dim=1)
       

        out = self.tdUp(drop5, drop4, x1, x2, x3)
        reversed_idx = list(reversed(range(out.shape[1])))
        # rev_index = list(reversed([i for i in range(out.size(1))]))
        # reversed_out = out[:,rev_index,...]

        # out = self.blstm2(out, reversed_out)
        out_rev = out[:,reversed_idx,...]
        # # out = torch.sum(out, dim=1)
        out = self.bcgru(out, out_rev)

        batch, timesteps, channels, height, width = out.size()

        out = out.view(timesteps*batch, channels, height, width)

        #final_out = out[:,-1,...]

        logits = self.outc(out)


        return logits

class SensorAblatedBLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()

        self.unetDown = unetDown(n_channels)
        self.unetUp = unetUp(n_classes, bilinear)
        
        self.tdDown = TimeDistributedDown(self.unetDown)
       # self.blstm1 = ConvBLSTM(in_channels=512, hidden_channels=1024, kernel_size=(3, 3), batch_first=True)
        self.tdUp = TimeDistributedUp(self.unetUp)
        self.blstm2 = ConvBLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)
        # self.bcgru = ConvBGRU(in_channels=64, hidden_channels=64, kernel_size=(3, 3), num_layers=1 ,batch_first=True)
      #  self.clstm = ConvLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3),batch_first=True)

        self.outc = OutConv(64, 1)

    def forward(self, x):

        drop5, drop4, x1, x2, x3 = self.tdDown(x)

       # print(f'the outsize is {drop4.size()}')

        # rev_index = list(reversed([i for i in range(drop4.size(1))]))
        # reversed_out = drop4[:,rev_index,...]

        # #print(f'rev size: {reversed_out.size()}')

        # out = self.blstm1(drop4, reversed_out)

        # batch, timesteps, channels, height, width = drop5.size()
        # drop5 = drop5.contiguous().view(batch*timesteps, channels, height, width) 

        out = self.tdUp(drop5, drop4, x1, x2, x3)

        # rev_index = list(reversed([i for i in range(out.size(1))]))
        # reversed_out = out[:,rev_index,...]

        # out = self.blstm2(out, reversed_out)

        batch, timesteps, channels, height, width = out.size()


        # # out = torch.sum(out, dim=1)
        out = self.blstm2(out[:,:timesteps//2+1,...], out[:,timesteps//2:,...])

        # final_out = out[:,-1,...]

        logits = self.outc(out[:,-1,...])


        return logits


class SensorAblatedLSTMLast(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()

        self.unetDown = unetDown(n_channels)
        self.unetUp = unetUp(n_classes, bilinear)
        
        self.tdDown = TimeDistributedDown(self.unetDown)
       # self.blstm1 = ConvBLSTM(in_channels=512, hidden_channels=1024, kernel_size=(3, 3), batch_first=True)
        self.tdUp = TimeDistributedUp(self.unetUp)
        # self.blstm2 = ConvBLSTM(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)
        # self.bcgru = ConvBGRU(in_channels=64, hidden_channels=64, kernel_size=(3, 3), num_layers=1 ,batch_first=True)
        self.clstm = ConvLSTM(in_channels=1, hidden_channels=1, kernel_size=(3, 3),batch_first=True)

        self.outc = OutConv(64, 1)

    def forward(self, x):

        drop5, drop4, x1, x2, x3 = self.tdDown(x)

       # print(f'the outsize is {drop4.size()}')

        # rev_index = list(reversed([i for i in range(drop4.size(1))]))
        # reversed_out = drop4[:,rev_index,...]

        # #print(f'rev size: {reversed_out.size()}')

        # out = self.blstm1(drop4, reversed_out)

        # batch, timesteps, channels, height, width = drop5.size()
        # drop5 = drop5.contiguous().view(batch*timesteps, channels, height, width) 

        out = self.tdUp(drop5, drop4, x1, x2, x3)

        # rev_index = list(reversed([i for i in range(out.size(1))]))
        # reversed_out = out[:,rev_index,...]

        # out = self.blstm2(out, reversed_out)

        # # out = torch.sum(out, dim=1)
        out, _ = self.clstm(out)

        #final_out = out[:,-1,...]

        logits = self.outc(out[:,-1,...].squeeze())


        return logits




