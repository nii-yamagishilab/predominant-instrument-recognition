import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'ThinResNet',
           'resnet18', 'resnet34', 'thin_resnet34', 'se_resnet34',
           'resnet50', 'resnet101', 'resnet152', 'resnet50_e1', 'shared_resnet34',
           'v_resnet34']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # F_squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ShareBB(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_layers=1):
        super(ShareBB, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes))
        self.conv_list = []
        self.n_layers = n_layers
        for i in range(1, n_layers):
            self.conv_list.append(
                nn.Sequential(
                    conv3x3(planes, planes),
                    nn.BatchNorm2d(planes),
                    nn.BatchNorm2d(planes)
                ))
        self.blocks = nn.Sequential(*self.conv_list)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.downsample_1 = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn1(out)

        if self.downsample_1 is not None:
            residual = self.downsample_1(x)
        out += residual
        out = self.relu(out)

        for ii in range(0, self.n_layers - 1):
            residual = out
            out = self.blocks[ii][0](out)
            out = self.blocks[ii][1](out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.blocks[ii][2](out)

            out += residual
            out = self.relu(out)

        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ThinResNet(nn.Module):
    """ResNet with smaller channel dimensions
    """

    def __init__(self, block, layers):
        self.inplanes = 8
        super(ThinResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((1, 3))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # x = self.maxpool(x)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)

        return x


class SharedResNet(nn.Module):

    def __init__(self, layers):
        self.inplanes = 16
        super(SharedResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ShareBB(16, 16, n_layers=layers[0])
        self.layer2 = ShareBB(
            16, 32, stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(16, 32,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32)),
            n_layers=layers[1]
        )
        self.layer3 = ShareBB(
            32, 64, stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(32, 64,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64)),
            n_layers=layers[2]
        )
        self.layer4 = ShareBB(
            64, 128, stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(64, 128,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128)),
            n_layers=layers[3]
        )
        self.avgpool = nn.AvgPool2d((1, 16))  # maybe this need to be changed to (1,16) for 122-bin CQT?
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # suppose input size (128, 250, 122) (BS, NFrames, NBins), model = resnet34 (shi+)
        # --> torch.Size([128, 1, 250, 122])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # --> torch.Size([128, 16, 250, 122])
        x = self.layer1(x)
        # --> torch.Size([128, 16, 250, 122])
        x = self.layer2(x)
        # --> torch.Size([128, 32, 125, 61])
        x = self.layer3(x)
        # --> torch.Size([128, 64, 63, 31])
        x = self.layer4(x)
        # --> torch.Size([128, 128, 32, 16])
        x = self.avgpool(x)
        # --> torch.Size([128, 128, 32, 1])
        x = x.squeeze(dim=3).permute(0, 2, 1)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((1, 16))  # maybe this need to be changed to (1,16) for 122-bin CQT?
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # suppose input size (128, 250, 122) (BS, NFrames, NBins), model = resnet34 (shi+)
        # --> torch.Size([128, 1, 250, 122])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # --> torch.Size([128, 16, 250, 122])
        x = self.layer1(x)
        # --> torch.Size([128, 16, 250, 122])
        x = self.layer2(x)
        # --> torch.Size([128, 32, 125, 61])
        x = self.layer3(x)
        # --> torch.Size([128, 64, 63, 31])
        x = self.layer4(x)
        # --> torch.Size([128, 128, 32, 16])
        x = self.avgpool(x)
        # --> torch.Size([128, 128, 32, 1])
        x = x.squeeze(dim=3).permute(0, 2, 1)
        return x

class V_ResNet(nn.Module):

    def __init__(self, block, layers, inplanes=16):
        self.inplanes = inplanes
        super(V_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, (inplanes * 2), layers[1], stride=2)
        self.layer3 = self._make_layer(block, (inplanes * 4), layers[2], stride=2)
        self.layer4 = self._make_layer(block, (inplanes * 8), layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((1, 16))  # maybe this need to be changed to (1,16) for 122-bin CQT?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # suppose input size (128, 250, 122) (BS, NFrames, NBins), model = resnet34 (shi+)
        # --> torch.Size([128, 1, 250, 122])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # --> torch.Size([128, 16, 250, 122])
        x = self.layer1(x)
        # --> torch.Size([128, 16, 250, 122])
        x = self.layer2(x)
        # --> torch.Size([128, 32, 125, 61])
        x = self.layer3(x)
        # --> torch.Size([128, 64, 63, 31])
        x = self.layer4(x)
        # --> torch.Size([128, 128, 32, 16])
        x = self.avgpool(x)
        # --> torch.Size([128, 128, 32, 1])
        x = x.squeeze(dim=3).permute(0, 2, 1)
        return x


def v_resnet34(layer_nums_txt):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param block_nums: block numbers
    """

    block_nums = list(map(int, layer_nums_txt.strip().split('-')))
    assert len(block_nums) == 5
    layers = block_nums[:-1]
    inplanes = block_nums[-1]
    model = V_ResNet(BasicBlock,layers, int(inplanes))
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def shared_resnet34(layer_nums_txt):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param block_nums: block numbers
    """

    block_nums = list(map(int, layer_nums_txt.strip().split('-')))
    assert len(block_nums) == 4
    model = SharedResNet(block_nums)
    return model


def thin_resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ThinResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet34(**kwargs):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_e1(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck_e1, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


class Bottleneck_e1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_e1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
