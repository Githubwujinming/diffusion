import itertools
import os
import PIL
import kornia
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
import pytorch_lightning as pl
from torchvision import transforms
from src.lr_scheduler import LambdaLinearScheduler
from torch.optim.lr_scheduler import LambdaLR

# from utils.lr_scheduler import LambdaLinearScheduler
from src.util import instantiate_from_config
from utils.SCD_metrics import RunningMetrics
from utils.scdloss import SCDLoss, softmax_mse_loss, consistency_weight
from utils.helpers import DeNormalize
import torch.optim.lr_scheduler as lrs
from src.data import Index2Color, Color2Index, TensorIndex2Color
from .pertubs import *

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
from torchvision import models
class FCN(nn.Module):
    def __init__(self, in_channels=3,  pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()        
        self.FCN = FCN(in_channels,pretrained=pretrained)
    
    def base_forward(self, x):
       
        x = self.FCN.layer0(x) #size:1/2
        x = self.FCN.maxpool(x) #size:1/4
        x = self.FCN.layer1(x) #size:1/4
        x = self.FCN.layer2(x) #size:1/8
        x = self.FCN.layer3(x) #size:1/8
        x = self.FCN.layer4(x) #size:1/8
        return x

    def forward(self, x1, x2):
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        return [x1, x2]


class SCDHead(nn.Module):
    def __init__(self, in_channel,num_classes=7) -> None:
        super().__init__()
        self.classifier1 = nn.Conv2d(in_channel, num_classes, kernel_size=1)
        
        self.classifier2 = nn.Conv2d(in_channel, num_classes, kernel_size=1)
        
        self.classifierCD = nn.Sequential(nn.Conv2d(in_channel, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
    
    def forward(self, x1, x2, change):
        change = self.classifierCD(change)
        x1 = self.classifier1(x1)
        x2 = self.classifier2(x2)       
        return x1, x2, change

class Neck(nn.Module):
    def __init__(self,embed_dim, mid_dim=128) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(embed_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(mid_dim), nn.ReLU())
        self.resCD = self._make_layer(ResBlock, mid_dim*2, mid_dim, 3, stride=1)
        
        
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )
        
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    def base_forward(self, x1, x2):
        x1 = self.head(x1)
        x2 = self.head(x2)
        change = [x1, x2]
        change = torch.cat(change, 1)
        change = self.resCD(change)
        return x1, x2, change

    def forward(self, x1, x2):
         return self.base_forward(x1, x2)
  

class SCDDecoder(nn.Module):
    def __init__(self, num_classes=7, mid_dim=128, embed_dim=1024, x_size=(512,512)) -> None:
        super().__init__()
        self.neck = Neck(embed_dim=embed_dim, mid_dim=mid_dim)
        self.head = SCDHead(mid_dim, num_classes)   
        self.x_size = x_size    
       
    def forward(self, x, perturbation = None, o_l = None):     
        if perturbation is not None:
            x = torch.cat(x, dim=1)
            x = perturbation(x, o_l)
            x1, x2 = torch.chunk(x, dim=1,chunks=2) 
        else:
            x1, x2 = x
        x1, x2, change = self.neck(x1, x2)
        x1, x2, change = self.head(x1, x2, change)
        return [F.interpolate(x1, self.x_size, mode='bilinear', align_corners=True), 
                F.interpolate(x2, self.x_size, mode='bilinear', align_corners=True), 
                F.interpolate(change, self.x_size, mode='bilinear', align_corners=True)]


from src.modules.attention.model import AttentionBlock


class SANeckAfter(Neck):
    def __init__(self, embed_dim, num_res_blocks=2, mid_dim=128, ks=3, stride=3, uf=1, df=1) -> None:
        super().__init__(embed_dim, mid_dim)
        self.ks = (ks, ks)
        self.stride = stride
        self.uf = uf
        self.df = df
        self.attn = nn.Sequential(*[AttentionBlock(mid_dim*2, 8) for i in range(num_res_blocks)])
        
    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
            """
            分成小块展开，
            :param x: img of size (bs, c, h, w)
            :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
            """
            bs, nc, h, w = x.shape

            if uf == 1 and df == 1:
                fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
                unfold = torch.nn.Unfold(**fold_params)

                fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            # 这是上采样
            elif uf > 1 and df == 1:
                fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
                unfold = torch.nn.Unfold(**fold_params)

                fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                    dilation=1, padding=0,
                                    stride=(stride[0] * uf, stride[1] * uf))
                fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            # 下采样
            elif df > 1 and uf == 1:
                fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
                unfold = torch.nn.Unfold(**fold_params)

                fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                    dilation=1, padding=0,
                                    stride=(stride[0] // df, stride[1] // df))
                fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            else:
                raise NotImplementedError

            return fold, unfold


    def patch_attn(self, x):
        fold, unfold = self.get_fold_unfold(x, self.ks, self.stride, self.uf, self.df)
        x = unfold(x) #
        x = x.view((x.shape[0], -1, self.ks[0], self.ks[1], x.shape[-1]))
        out_list = [self.attn(x[:, :, :, :, i]) for i in range(x.shape[-1])]
        x = torch.stack(out_list, dim=-1)
        x = x.view((x.shape[0], -1, x.shape[-1]))
        x = fold(x)
        return x
        
    def forward(self, x1, x2):
        x1 = self.head(x1)
        x2 = self.head(x2)
        x = [x1, x2]
        x = torch.cat(x, 1)
        x = self.patch_attn(x)
        x1, x2 = torch.chunk(x, dim=1, chunks=2)
        change = self.resCD(x)
        
        return x1, x2, change
        

class SANeckBefore(SANeckAfter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        num_res_blocks = kwargs['num_res_blocks'] if hasattr(kwargs, 'num_res_blocks') else 2
        self.attn = nn.Sequential(*[AttentionBlock(1024, 32) for i in range(num_res_blocks)])
    
        
        
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        x = self.patch_attn(x)
        x1, x2 = torch.chunk(x, dim=1, chunks=2)
        x1 = self.head(x1)
        x2 = self.head(x2)
        change = torch.cat([x1, x2], 1)
        change = self.resCD(change)
        return x1, x2, change
 
class DecoderFromConfig(nn.Module):
    def __init__(self, neck_config, head_config, x_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)   
        self.neck = instantiate_from_config(neck_config)
        self.head = instantiate_from_config(head_config)
        self.x_size = x_size    
       
    def forward(self, x, perturbation = None, o_l = None):     
        if perturbation is not None:
            x = torch.cat(x, dim=1)
            x = perturbation(x, o_l)
            x1, x2 = torch.chunk(x, dim=1,chunks=2) 
        else:
            x1, x2 = x
        x1, x2, change = self.neck(x1, x2)
        x1, x2, change = self.head(x1, x2, change)
        return [F.interpolate(x1, self.x_size, mode='bilinear', align_corners=True), 
                F.interpolate(x2, self.x_size, mode='bilinear', align_corners=True), 
                F.interpolate(change, self.x_size, mode='bilinear', align_corners=True)]

     
         
    
# from src.modules.attention.spatial_attention import SpatialTransformer
# class CANeckAfter(SANeckAfter):
#     def __init__(self, mid_dim=128, *args, **kwargs):
#         super().__init__( *args, **kwargs)
#         self.attn = SpatialTransformer(mid_dim*2, n_heads=8, d_head=16, depth=2)
    
#     def patch_attn(self, x):
#         fold, unfold = self.get_fold_unfold(x, self.ks, self.stride, self.uf, self.df)
#         x = unfold(x) #
#         x = x.view((x.shape[0], -1, self.ks[0], self.ks[1], x.shape[-1]))
#         out_list = [self.attn(x[:, :, :, :, i]) for i in range(x.shape[-1])]
#         x = torch.stack(out_list, dim=-1)
#         x = x.view((x.shape[0], -1, x.shape[-1]))
#         x = fold(x)
#         return x
        
#     def forward(self, x1, x2):
#         x1 = self.head(x1)
#         x2 = self.head(x2)
#         x = [x1, x2]
#         x = torch.cat(x, 1)
#         change = self.resCD(x)
#         x = self.patch_attn(x, change)
#         x1, x2 = torch.chunk(x, dim=1, chunks=2)
#         change = self.resCD(x)
        
#         return x1, x2, change   
    
# class CANeckBefore(SANeckBefore):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__( *args, **kwargs)
#         self.attn = SpatialTransformer(1024, n_heads=32, d_head=32, depth=2)
        