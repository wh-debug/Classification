import os
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models
import wandb

from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from mobilenetv1 import  MobileNetV1
import torch.nn.functional as F


wandb.login(key="c0f5a585df67a24a03da866c96a0833d637d49ed")
wandb.init(project="dog-vs-cat-20", entity="daylight2", save_code=True)

# 超参数设置
wandb.config = {
  "learning_rate": 0.0001,
  "epochs": 30,
  "batch_size": 8
}

batch_size = 8

# 图像增强
dataset_transform = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),  # 随机反转-45到45度之间
        transforms.Resize(300),
        transforms.CenterCrop(224),  # 从中心开始裁剪224
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转， 选择一个概率
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        # 参数1为亮度， 参数2为对比度， 参数3为饱和度， 参数4为色相
        transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率， 3通道是R G B
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 均值 标准差
    ]),

    'vaild': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),

    'test': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.filelength = len(file_list)

    def __len__(self):
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0
        return img_transformed, label


train_list = glob.glob(os.path.join("/home/wh/vscode/dog_vs_vat/train", '*.jpg'))
test_list = glob.glob(os.path.join("/home/wh/vscode/dog_vs_vat/test", '*.jpg'))

labels = [path.split('/')[-1].split('.')[0] for path in train_list]

train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=0)

train_data = CatsDogsDataset(train_list, transform=dataset_transform['train'])
valid_data = CatsDogsDataset(valid_list, transform=dataset_transform['vaild'])
test_data = CatsDogsDataset(test_list, transform=dataset_transform['test'])


train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
#
# Alexnet网络
model = torchvision.models.alexnet(pretrained=True)
model.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=9216, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=4096, out_features=1000, bias=True),
    nn.Linear(in_features=1000, out_features=2, bias=True),
)

model = model.cuda()
num_epochs = 20
lr = 1e-3


criterion = nn.CrossEntropyLoss() # 使用pytorch中的二元交叉熵函数不需要使用softmax, 输入是[feature_map, C], [feature_map]
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

num = 0

for epoch in range(num_epochs):
    model.train()
    for X, y in train_loader:
        X = X.cuda()
        y = y.cuda()  # 标签

        optimizer.zero_grad()
        prediction = model(X)
        print(prediction)
        print(y)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        num = num + 1
        _, predictions = torch.max(prediction, 1)
        if num == 5:
            print('Epoch [{}/{}],  Loss: {:.4f}%'.format(epoch + 1, num_epochs, loss.item()))
            num = 0
        # wandb.log({
        #     'mlp/train_loss': loss.item(),
        #     'mlp/train_accuracy': (y == predictions).sum() / len(y),
        # })

    with torch.no_grad():
        model.eval()
        for X, y in valid_loader:
            X = X.cuda()
            y = y.cuda()

            prediction = model(X)

            loss = criterion(prediction, y)
            _, predictions = torch.max(prediction, 1)

            wandb.log({
                'mlp/val_loss': loss.item(),
                'mlp/val_accuracy': (y == predictions).sum() / len(y),
            })