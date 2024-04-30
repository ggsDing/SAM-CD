import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, List
from utils.misc import initialize_weights
from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam
import os
working_path = os.path.abspath('.')

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def build_efficient_sam_vitt():    
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint = working_path+'/EfficientSAM/weights/efficient_sam_vitt.pt',
    ).eval()


def build_efficient_sam_vits():
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint = working_path+'/EfficientSAM/weights/efficient_sam_vits.pt',
    ).eval()


class Space_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Space_Attention, self).__init__()
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()        
        A = self.SA(x)
        return A


class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
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


class SAM_CD(nn.Module):
    def __init__(self, num_embed=8):
        super(SAM_CD, self).__init__()
        self.sam = build_efficient_sam_vitt()
                
        self.Adapter = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU())    
        self.segmenter = nn.Conv2d(64, num_embed, kernel_size=1)
            
        self.SA = Space_Attention(num_embed*2, 16, 4)
        self.resCD = self._make_layer(ResBlock, 128, 128, 6, stride=1)
        self.headC = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.segmenterC = nn.Conv2d(16, 1, kernel_size=1)
                                        
        for param in self.sam.parameters():
            param.requires_grad = False
        initialize_weights(self.Adapter, self.segmenter, self.resCD, self.headC, self.segmenterC)

    def run_encoder(self, image):
        image_embeddings = self.sam.get_image_embeddings(image)
        return image_embeddings

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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
    
        input_shape = x1.shape[-2:]
        featsA = self.run_encoder(x1)
        featsB = self.run_encoder(x2)
        
        featA_adpt = self.Adapter(featsA.clone())
        featB_adpt = self.Adapter(featsB.clone())        
        outA = self.segmenter(featA_adpt)
        outB = self.segmenter(featB_adpt)
             
        A = self.SA(torch.cat([outA, outB], dim=1))  
        featC = torch.cat([featA_adpt, featB_adpt], 1)
        featC = self.resCD(featC)
        featC = self.headC(featC) * A
        outC = self.segmenterC(featC)
        
        return F.interpolate(outC, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(outA, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(outB, input_shape, mode="bilinear", align_corners=True)