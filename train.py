# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
import torchvision

from path import IMAGE_PATH
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
from model import Model
from torch.utils.data import DataLoader
from path import MODEL_PATH

cuda_avail = torch.cuda.is_available()

EPOCHS = 1
BATCH = 32
OVERWRITE_MODEL = True
PRETRAINED_MODEL = True
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
imageSet = ImageFolder(IMAGE_PATH, transform=transform)
data = DataLoader(imageSet, batch_size=BATCH, shuffle=True)


def eval(model, x_test, y_test):
    cnn.eval()
    batch_eval = model.batch_iter(x_test, y_test)
    total_acc = 0.0
    data_len = len(x_test)  # 数据长度
    for x_batch, y_batch in batch_eval:
        # x_batch.shape: [32,3,224,224]
        # y_batch.shape: [32]
        batch_len = len(x_batch)
        outputs = cnn(x_batch)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
    return total_acc / data_len


cnn = torchvision.models.densenet121(pretrained=True)  # 加载预训练模型
for param in cnn.parameters():
    param.requires_grad = False
num_features = cnn.classifier.in_features
cnn.classifier = nn.Linear(num_features, 256)
# Note that Caltech256 have 257 classes. So if you want to classify it, please set classes as 257 or delete a class.

if cuda_avail:
    cnn.cuda()

optimizer = Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.999))  # 优化器Adam
loss_fn = nn.CrossEntropyLoss()  # loss function with Cross




# 训练并评估模型
model = Model(data)
best_accuracy = 0
for i in range(EPOCHS):
    cnn.train()
    for j, trainData in enumerate(data):
        x_train, y_train = trainData
        x_train = x_train.float()

        if cuda_avail:
            x_train = Variable(x_train.cuda())
            y_train = Variable(y_train.cuda())

        outputs = cnn(x_train)
        _, prediction = torch.max(outputs.data, 1)

        optimizer.zero_grad()

        loss = loss_fn(outputs, y_train.long())
        loss.backward()
        optimizer.step()
        if j % 10 == 0:  # trian : test == 9:1
            x_test = x_train
            y_test = y_train
            if cuda_avail:
                x_test = Variable(x_test.cuda())
                y_test = Variable(y_test.cuda())
            train_accuracy = eval(model, x_test, y_test)
            if train_accuracy > best_accuracy:
                best_accuracy = train_accuracy
                model.save_model(cnn, MODEL_PATH, overwrite=OVERWRITE_MODEL)
                print("epoch %d step %d, best accuracy %g" % (i, j, best_accuracy))
        print("epoch %d step %d" % (i, j))

    print(str(i) + "/" + str(EPOCHS))

print("Training is over, best accuracy is %g" % best_accuracy)
