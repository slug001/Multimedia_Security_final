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

    def predict_internal_states(self, x):
        internals = []
        # conv1->bn1->relu
        y = self.conv1(x); internals.append(y.clone().detach())
        y = self.bn1(y); y = F.relu(y)
        # conv2->bn2
        y = self.conv2(y); internals.append(y.clone().detach())
        y = self.bn2(y)
        # shortcut
        sc = self.shortcut(x)
        # sum + relu
        y = F.relu(y + sc)
        return internals, y


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

    def predict_internal_states(self, x):
        internals=[]
        # conv1
        y = self.conv1(x); internals.append(y.clone().detach())
        y = self.bn1(y); y = F.relu(y)
        # conv2
        y = self.conv2(y); internals.append(y.clone().detach())
        y = self.bn2(y); y = F.relu(y)
        # conv3
        y = self.conv3(y); internals.append(y.clone().detach())
        y = self.bn3(y)
        # shortcut + sum + relu
        sc = self.shortcut(x)
        y = F.relu(y + sc)
        return internals, y


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

class ResNetWithInternals(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        # 定義與原 ResNet 相同
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers=[]
        strides=[stride]+[1]*(num_blocks-1)
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)

    def predict_internal_states(self, x):
        internals=[]
        # 第 1 層 conv1
        y = self.conv1(x); internals.append(y.clone().detach())
        y = self.bn1(y); y = F.relu(y)
        # 依序跑每個 block 並收集它裡面的 internals
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                b_ints, y = block.predict_internal_states(y)
                internals += b_ints
        # 池化
        y = F.avg_pool2d(y, 4)
        # 展平
        y = y.view(y.size(0), -1)
        # 最後一層線性
        z = self.linear(y); internals.append(z.clone().detach())
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

if __name__ == '__main__':
    # test 1
    x = torch.randn(1, 3, 32, 32)
    model1 = ResNet(BasicBlock, [2,2,2,2])
    model2 = ResNetWithInternals(BasicBlock, [2,2,2,2])
    
    model2.load_state_dict(model1.state_dict())

    out1 = model1(x)
    out2 = model2(x)
    print(torch.allclose(out1, out2))  # Should be True

    # test 2
    for seed in [42, 123, 999]:
        torch.manual_seed(seed)
        for _ in range(10):                # 10 個 batch
            x = torch.randn(8, 3, 32, 32)  # batch size 8
            out1 = model1(x)
            out2 = model2(x)
            assert torch.allclose(out1, out2, atol=1e-6)
    print("Random batch test passed.")

    # test 3
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=16, shuffle=False)

    # 拿前 5 個 batch：
    for i, (x, _) in enumerate(loader):
        if i >= 5: break
        out1 = model1(x)
        out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-6)
    print("CIFAR-10 batch test passed.")
    
    # test 4
    x = torch.randn(2, 3, 32, 32)
    model = ResNetWithInternals(BasicBlock, [2,2,2,2])
    # 如果需要比兩個models
    # model2.load_state_dict(model1.state_dict())

    # 1. 呼叫 predict_internal_states
    internals = model.predict_internal_states(x)
    print(f"Number of internal states: {len(internals)}")

    # 2. 印出每個中間層 shape
    for i, t in enumerate(internals):
        print(f"Internal {i}: shape={t.shape}")

    # 3. 最後一個 internal 應該和 forward 結果一致
    out = model(x)
    last_internal = internals[-1]
    print("Forward output == last internal:", torch.allclose(out, last_internal, atol=1e-6))