from torch import nn


class bottleneck(nn.Module):
    def __init__(self, input, output, stride, expand_ratio):
        super(bottleneck, self).__init__()
        self.stride = stride
        # 中间扩展层的通道数
        hidden_dim = round(input * expand_ratio)
        self.conv = nn.Sequential(
            # 1 x 1 逐点卷积进行升维
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU6(inplace=True),
            # 深度可分离的模块
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 1 x 1 的逐点卷积进行降维
            nn.Conv2d(in_channels=hidden_dim, out_channels=output, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output),
        )

    def forward(self, x):
        return x + self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()



if __name__ == "__main__":
    block = MobileNetV2(input=24, output=24, stride=1, expand_ratio=6).cuda()
    print(block)
