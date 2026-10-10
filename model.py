import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet для CIFAR как у авторов из файла Resnet56Cifar.prototxt.


class Block(nn.Module): # блок ResNet, при downsample=True карта уменьшается в 2 раза
    def __init__(self, channels, downsample=False, bn_momentum=0.1, factor=2):
        super().__init__()
        self.downsample = downsample
        self.factor = factor
        self.conv1 = nn.Conv2d(channels, channels, 3, 2 if downsample else 1, 1, bias=False) # в блоке с уменьшением stride 2, но число каналов не меняется, так у авторов
        self.bn1 = nn.BatchNorm2d(channels, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=bn_momentum)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            x = F.avg_pool2d(x, 3, 2, ceil_mode=True) # shortcut тоже уменьшаем, через avg-pool 3x3, как у авторов
        out = F.relu(out + x)
        if self.downsample:
            out = torch.cat([out] + [torch.zeros_like(out)] * (self.factor - 1), dim=1) # после сложения дописываем нулевые каналы: 16 -> 32 -> 64, у wide 16 -> 64 -> 256
        return out


class ResNet(nn.Module):
    def __init__(self, depth=56, num_classes=10, bn_momentum=0.1, factor=2):
        super().__init__()
        n = (depth - 2) // 6 # сколько блоков в каждой из 3 стадий, у ResNet-56 это 9
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16, momentum=bn_momentum)

        widths = [16, 16 * factor, 16 * factor ** 2] # каналы по стадиям, factor=4 это wide ResNet, мое прочтение фразы из статьи
        blocks = []
        for stage, channels in enumerate(widths):
            for i in range(n):
                if stage > 0 and i == 0:
                    blocks.append(Block(channels // factor, downsample=True, bn_momentum=bn_momentum, factor=factor))
                else:
                    blocks.append(Block(channels, bn_momentum=bn_momentum))
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(widths[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# DenseNet-40 (k = 12) как в github.com/liuzhuang13/DenseNetCaffe


def bn_relu_conv(cin, cout, kernel, dropout, bn_momentum):
    return nn.Sequential(nn.BatchNorm2d(cin, momentum=bn_momentum), nn.ReLU(),
                         nn.Conv2d(cin, cout, kernel, 1, kernel // 2, bias=False), nn.Dropout(dropout))


class DenseLayer(nn.Module):
    def __init__(self, channels, growth, dropout, bn_momentum):
        super().__init__()
        self.f = bn_relu_conv(channels, growth, 3, dropout, bn_momentum)

    def forward(self, x):
        return torch.cat([x, self.f(x)], dim=1) # новые признаки дописываем к старым, в этом суть DenseNet


class DenseNet(nn.Module):
    def __init__(self, depth=40, growth=12, num_classes=10, dropout=0.2, bn_momentum=0.1):
        super().__init__()
        n = (depth - 4) // 3
        channels = 16
        layers = [nn.Conv2d(3, channels, 3, 1, 1, bias=False)]
        for stage in range(3):
            for _ in range(n):
                layers.append(DenseLayer(channels, growth, dropout, bn_momentum))
                channels += growth
            if stage < 2:
                layers += [bn_relu_conv(channels, channels, 1, dropout, bn_momentum), nn.AvgPool2d(2)] # переход между блоками: свертка 1x1 и avg-pool, карта уменьшается в 2 раза
        layers += [nn.BatchNorm2d(channels, momentum=bn_momentum), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)
        return self.fc(x)


class LeNet(nn.Module): # LeNet как в Caffe
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)
