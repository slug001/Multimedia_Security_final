#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def get_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class SequentialWithInternalStatePrediction(nn.Sequential):
    def predict_internal_states(self, x):
        internals = []
        for module in self:
            x = module(x)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                internals.append(x.clone().detach())
        return internals, x

class ResNetWithInternals(ResNet):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__(block, num_blocks, num_classes)
        # 把 conv1+bn1+relu 包成可回傳中間態的序列
        self.initial = SequentialWithInternalStatePrediction(
            self.conv1, self.bn1, nn.ReLU(inplace=True)
        )
        # 同理，將每層 layer1~4 裏的 block, 再用一個 wrapper 包一次
        self.layer1 = SequentialWithInternalStatePrediction(*self.layer1)
        self.layer2 = SequentialWithInternalStatePrediction(*self.layer2)
        self.layer3 = SequentialWithInternalStatePrediction(*self.layer3)
        self.layer4 = SequentialWithInternalStatePrediction(*self.layer4)
        # 線性分類器也一樣
        self.classifier = SequentialWithInternalStatePrediction(
            nn.Flatten(),
            self.linear
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict_internal_states(self, x):
        internals = []
        out, x = self.initial.predict_internal_states(x)
        internals += out
        out, x = self.layer1.predict_internal_states(x)
        internals += out
        out, x = self.layer2.predict_internal_states(x)
        internals += out
        out, x = self.layer3.predict_internal_states(x)
        internals += out
        out, x = self.layer4.predict_internal_states(x)
        internals += out
        # 池化與扁平化後再收一次
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        out, x = self.classifier.predict_internal_states(x)
        internals += out
        return internals

# def NarrowResNet18():
#     return NarrowResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18WithInternals():
    return ResNetWithInternals(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
# class NarrowResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(NarrowResNet, self).__init__()
#         self.in_planes = 1

#         self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(1)
#         self.layer1 = self._make_layer(block, 1, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 1, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 1, num_blocks[2], stride=2)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         return out

class NarrowResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(NarrowResNet, self).__init__()
        self.in_planes = 1

        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.layer1 = self._make_layer(block, 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 1, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 1, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 1, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def NarrowResNet18():
    return NarrowResNet(BasicBlock, [2, 2, 2, 2])
    
# class narrow_ResNet(nn.Module):
#     # by default : block = BasicBlock
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(narrow_ResNet, self).__init__()

#         self.in_planes = 1 # one channel chain

#         self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False) # original num_channel = 16 
#         self.bn1 = nn.BatchNorm2d(1) # bn1
#         # => 1 x 32 x 32

#         self.layer1 = self._make_layer(block, 1, num_blocks[0], stride=1) # original num_channel = 16 
#         # => 1 x 32 x 32

#         self.layer2 = self._make_layer(block, 1, num_blocks[1], stride=2) # original num_channel = 32
#         # => 1 x 16 x 16

#         self.layer3 = self._make_layer(block, 1, num_blocks[2], stride=2) # original num_channel = 64
#         # => 1 x 8 x 8

#         self.apply(_weights_init)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
        
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         return out

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_feature(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

def get_model(data):
    if data == 'fmnist' or data == 'fedemnist':
        return CNN_MNIST()
    elif data == 'cifar10':
        return CNN_CIFAR()
               

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        #self.drop1 = nn.Dropout2d(p=0.5)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(9216, 128)
        #self.drop2 = nn.Dropout2d(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x        
    
    def get_feature(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return x  


#### checking if ResNet weights are identical to ResNetWithInternals weights

def transfer_resnet_weights_to_internals(resnet, resnet_internals):
    # Copy initial
    resnet_internals.initial[0].load_state_dict(resnet.conv1.state_dict())
    resnet_internals.initial[1].load_state_dict(resnet.bn1.state_dict())
    # Copy layers
    for i in range(1, 5):
        getattr(resnet_internals, f'layer{i}').load_state_dict(getattr(resnet, f'layer{i}').state_dict())
    # Copy classifier
    resnet_internals.classifier[1].load_state_dict(resnet.linear.state_dict())

if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    model1 = ResNet(BasicBlock, [2,2,2,2])
    model2 = ResNetWithInternals(BasicBlock, [2,2,2,2])
    transfer_resnet_weights_to_internals(model1, model2)

    out1 = model1(x)
    out2 = model2(x)
    print(torch.allclose(out1, out2))  # Should be True