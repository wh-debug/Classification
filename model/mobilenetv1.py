import torch
from torch import nn


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        # 标准卷积
        def conv_bn(dim_in, dim_out, stride):
            return  nn.Sequential(
                nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True)
            )

        def conv_dw(dim_in, dim_out, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=3, stride=stride, padding=1, groups=dim_in, bias=False),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True)
            )

        self .mobile = nn.Sequential(
            conv_bn(dim_in=3, dim_out=32, stride=2),
            conv_dw(dim_in=32, dim_out=64, stride=1),
            conv_dw(dim_in=64, dim_out=128, stride=2),
            conv_dw(dim_in=128, dim_out=128, stride=1),
            conv_dw(dim_in=128, dim_out=256, stride=2),
            conv_dw(dim_in=256, dim_out=256, stride=1),
            conv_dw(dim_in=256, dim_out=512, stride=2),
            conv_dw(dim_in=512, dim_out=512, stride=1),
            conv_dw(dim_in=512, dim_out=512, stride=1),
            conv_dw(dim_in=512, dim_out=512, stride=1),
            conv_dw(dim_in=512, dim_out=512, stride=1),
            conv_dw(dim_in=512, dim_out=512, stride=1),
            conv_dw(dim_in=512, dim_out=1024, stride=2),
            conv_dw(dim_in=1024, dim_out=1024, stride=1),
            nn.AvgPool2d(7),
        )
        self.Linear = nn.Linear(in_features=1024, out_features=2, bias=True)

    def forward(self, x):
        x = self.mobile(x)
        x = x.view(-1, 1024)
        x = self.Linear(x)
        return x


if __name__ == "__main__":
    mobilenet = MobileNetV1().cuda()
    input = torch.randn(8, 3, 224, 224).cuda()
    output = mobilenet(input)
    print(output.shape)
    print(torch.sum(output))
    print(f"max : {torch.max(output)}")
    print(f"min : {torch.min(output)}")

