# -*- coding: utf-8 -*
import numpy
import os
import torch
from torch.autograd import Variable
from path import MODEL_PATH

Torch_MODEL_NAME = "model.pkl"

cuda_avail = torch.cuda.is_available()


class Model:
    def __init__(self, data):
        self.data = data

    def predict_all(self):
        # 预测数据集中的全部的数据
        cnn = torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
        if cuda_avail:
            cnn.cuda()
        labelsRecord = []
        correctSum = 0
        dataSum = 0
        for i, trainData in enumerate(self.data):
            x_data, labels = trainData
            x_data = x_data.float()
            if cuda_avail:
                x_data = Variable(x_data.cuda())
            outputs = cnn(x_data)
            outputs = outputs.cpu()
            _, prediction = torch.max(outputs.data, 1)
            correct = (prediction == labels).sum().item()
            correctSum += correct
            dataSum += len(x_data)
            for tensorLabel in list(prediction):
                labelsRecord.append(tensorLabel.item())
            print('correct: %d, amount: %d, accuracy: %g' % (correctSum, dataSum, float(correctSum / dataSum)))

        return correctSum, dataSum, labelsRecord

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=Torch_MODEL_NAME, overwrite=False):
        self.check(path, overwrite)
        torch.save(network, os.path.join(path, name))

    def check(self, path, overwrite=False):  # overwrite model or not
        if overwrite:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        if not os.path.exists(path):
            os.makedirs(path)
