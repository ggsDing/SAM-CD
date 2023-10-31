import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class res34(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(res34, self).__init__()
        resnet = models.resnet34(pretrained=True)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, :3, :, :].copy_(resnet.conv1.weight.data[:, :3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU())

class res18(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(res18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, :3, :, :].copy_(resnet.conv1.weight.data[:, :3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU())

class ResNet_CD(nn.Module):
    def __init__(self, in_channels=6, num_classes=1, pretrained=True):
        super(ResNet_CD, self).__init__()
        self.ResNet = res18(in_channels, num_classes)                
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, t1, t2):
        input_size = t1.size()
        imgs = torch.cat([t1, t2], dim=1)
        
        x = self.ResNet.layer0(imgs)
        x = self.ResNet.maxpool(x)
        x = self.ResNet.layer1(x)
        x = self.ResNet.layer2(x)
        x = self.ResNet.layer3(x)
        x = self.ResNet.layer4(x)
        x = self.ResNet.head(x)
                
        out = self.classifier(x)
                
        return F.upsample(out, input_size[2:], mode='bilinear')
