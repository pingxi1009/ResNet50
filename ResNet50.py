import torch
from torch import nn
import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)       # 数据归一化处理，使其均值为0，方差为1，可有效避免梯度消失

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)       # 数据归一化处理，使其均值为0，方差为1，可有效避免梯度消失

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)   # 数据归一化处理，使其均值为0，方差为1，可有效避免梯度消失

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

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 遍历所有模块，然后对其中参数进行初始化
        for m in self.modules():    # self.modules()采用深度优先遍历的方式，存储了net的所有模块
            if isinstance(m, nn.Conv2d):        # 判断是不是卷积
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # 对权值参数初始化
            elif isinstance(m, nn.BatchNorm2d): # 判断是不是数据归一化
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 当是 3 4 6 3 的第一层时，由于跨层要做一个步长为 2 的卷积 size会变成二分之一，所以此处跳连接 x 必须也是相同维度
            downsample = nn.Sequential(     # 对跳连接 x 做 1x1 步长为 2 的卷积，保证跳连接的时候 size 一致
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # 将跨区的第一个要做步长为 2 的卷积添加到layer里面
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):                                      # 将除去第一个的剩下的 block 添加到layer里面
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained=False):
    '''Constructs a ResNet-50 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)
    # if pretrained:    # 加载已经生成的模型
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model