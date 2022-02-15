import torch
from torch import nn


class Fire(nn.Module):
    def __init__(self, input_channels, squeeze_planes, expand_planes):
        super(Fire, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=squeeze_planes, out_channels=expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(in_channels=squeeze_planes, out_channels=expand_planes, kernel_size=3, stride=1,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        # cat处理
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)

        return out


class SqueezeNet_1(nn.Module):
    def __init__(self):
        super(SqueezeNet_1, self).__init__()
        self.squeezenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(input_channels=96, squeeze_planes=16, expand_planes=64),
            Fire(input_channels=128, squeeze_planes=16, expand_planes=64),
            Fire(input_channels=128, squeeze_planes=32, expand_planes=128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(input_channels=256, squeeze_planes=32, expand_planes=128),
            Fire(input_channels=256, squeeze_planes=48, expand_planes=192),
            Fire(input_channels=384, squeeze_planes=48, expand_planes=192),
            Fire(input_channels=384, squeeze_planes=64, expand_planes=256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(input_channels=512, squeeze_planes=64, expand_planes=256),
            nn.Conv2d(in_channels=512, out_channels=1000, kernel_size=1, stride=1),
            nn.AvgPool2d(kernel_size=12, stride=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.squeezenet(x)

        return x


class SqueezeNet_2(nn.Module):
    def __init__(self):
        super(SqueezeNet_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire1 = Fire(input_channels=96, squeeze_planes=16, expand_planes=64)
        self.fire2 = Fire(input_channels=128, squeeze_planes=16, expand_planes=64)
        self.fire3 = Fire(input_channels=256, squeeze_planes=32, expand_planes=128)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire4 = Fire(input_channels=256, squeeze_planes=32, expand_planes=128)
        self.fire5 = Fire(input_channels=512, squeeze_planes=48, expand_planes=192)
        self.fire6 = Fire(input_channels=384, squeeze_planes=48, expand_planes=192)
        self.fire7 = Fire(input_channels=768, squeeze_planes=64, expand_planes=256)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire8 = Fire(input_channels=512, squeeze_planes=64, expand_planes=256)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1)
        self.avarge_pool = nn.AvgPool2d(kernel_size=12, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool_1(x)
        res1 = self.fire1(x)   # 96
        x = self.fire2(res1)   # 128
        x = torch.cat((res1, x), dim=1)
        x = self.fire3(x)
        res2 = self.max_pool_2(x)
        x = self.fire4(res2)
        x = torch.cat((res2, x), dim=1)
        res3 = self.fire5(x)
        x = self.fire6(res3)
        x = torch.cat((res3, x), dim=1)
        x = self.fire7(x)
        res4 = self.max_pool_3(x)
        x = self.fire8(res4)
        x = torch.cat((res4, x), dim=1)
        x = self.conv2(x)
        x = self.avarge_pool(x)
        x = self.softmax(x)

        return x


class SqueezeNet_3(nn.Module):
    def __init__(self):
        super(SqueezeNet_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire1 = Fire(input_channels=96, squeeze_planes=16, expand_planes=64)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=1, stride=1)
        self.fire2 = Fire(input_channels=256, squeeze_planes=16, expand_planes=64)
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1, stride=1)
        self.fire3 = Fire(input_channels=384, squeeze_planes=32, expand_planes=128)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire4 = Fire(input_channels=512, squeeze_planes=32, expand_planes=128)
        self.conv4 = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=1, stride=1)
        self.fire5 = Fire(input_channels=768, squeeze_planes=48, expand_planes=192)
        self.fire6 = Fire(input_channels=768, squeeze_planes=48, expand_planes=192)
        self.conv5 = nn.Conv2d(in_channels=1152, out_channels=512, kernel_size=1, stride=1)
        self.fire7 = Fire(input_channels=1152, squeeze_planes=64, expand_planes=256)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire8 = Fire(input_channels=1024, squeeze_planes=64, expand_planes=256)
        self.conv6 = nn.Conv2d(in_channels=1536, out_channels=1000, kernel_size=1, stride=1)
        self.avarge_pool = nn.AvgPool2d(kernel_size=12, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool_1(x)
        res = self.conv2(x)
        x = self.fire1(x)   # 96输入，经过地一个fire是128
        x = torch.cat((res, x), dim=1)  # 第一个cat点
        res = self.fire2(x)   # 128
        x = torch.cat((x, res), dim=1)
        res = self.conv3(x)
        x = self.fire3(x)
        x = torch.cat((res, x), dim=1)
        res = self.max_pool_2(x)
        x = self.fire4(res)
        x = torch.cat((res, x), dim=1)
        res = self.conv4(x)
        x = self.fire5(x)
        x = torch.cat((res, x), dim=1)
        res = self.fire6(x)
        x = torch.cat((x, res), dim=1)
        res = self.conv5(x)
        x = self.fire7(x)
        x = torch.cat((res, x), dim=1)
        res = self.max_pool_3(x)
        x = self.fire8(res)
        x = torch.cat((res, x), dim=1)
        x = self.conv6(x)
        x = self.avarge_pool(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    test = torch.randn(10, 3, 224, 224).cuda()
    model = SqueezeNet_1().cuda()
    fire_block = model(test)
    print(fire_block.shape)
    print(torch.sum(fire_block))
    print(f"max : {torch.max(fire_block)}")
    print(f"min : {torch.min(fire_block)}")
