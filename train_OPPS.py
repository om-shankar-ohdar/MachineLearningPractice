"""
main fle to execute
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import logging
import numpy as np

import argparse
import network

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch_size', default=128)
parser.add_argument('-lr', '--learning_rate', default=1e-3)
parser.add_argument('-e', '--epoch', default=200)
parser.add_argument('-tb', '--test_batch_size', default=128)
parser.add_argument('-r', '--root', required=True)


class Train:
    train_loader = None
    test_loader = None
    device = None
    cnnNet = None

    def __init__(self):
        cnnNet = network.Net()
        cnnNet.apply(self.init_weights)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def count_parameters(cls, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @classmethod
    def init_weights(cls, netLayer):
        if type(netLayer) == nn.Linear:
            torch.nn.init.xavier_uniform_(netLayer.weight)
            netLayer.bias.data.fill_(0.01)

    @classmethod
    def one_hot(cls, x, K):
        return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

    @classmethod
    def inf_generator(cls, iterable):
        """Allows training with DataLoaders in a single infinite loop:
            for i, (x, y) in enumerate(inf_generator(train_loader)):
        """
        iterator = iterable.__iter__()
        while True:
            try:
                yield iterator.__next__()
            except StopIteration:
                iterator = iterable.__iter__()

    @classmethod
    def accuracy(cls, model, dataset_loader):
        total_correct = 0
        for x, y in dataset_loader:
            x = x.view(-1, 28, 28, 1)
            x = torch.transpose(x, 1, 2)

            x = x.to(cls.device)
            y = cls.one_hot(np.array(y.numpy()), 47)

            target_class = np.argmax(y, axis=1)
            predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
            total_correct += np.sum(predicted_class == target_class)
        return total_correct / len(dataset_loader.dataset)

    @classmethod
    def loader(cls):

        # Tensor Transformers
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                # transforms.Normalize((mean,), (std,)),
            ]
        )

        transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((mean,), (std,)),
            ]
        )

        # Preparing Data Loaders
        train = datasets.EMNIST(
            root, split="balanced", train=True, download=True, transform=transform_train
        )
        test = datasets.EMNIST(
            root, split="balanced", train=False, download=True, transform=transform_valid
        )

        cls.train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
        )

        cls.test_loader = torch.utils.data.DataLoader(
            test, batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
        )

    @classmethod
    def training(cls):
        logging.basicConfig(filename="cnn2.log", level=logging.DEBUG)
        logging.info(cls.cnnNet)
        logging.info("Number of parameters: {}".format(cls.count_parameters(cls.cnnNet)))

        data_gen = cls.inf_generator(cls.train_loader)
        batches_per_epoch = len(cls.train_loader)

        optimizer = torch.optim.SGD(cls.cnnNet.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss().to(cls.device)

        best_acc = 0
        cls.cnnNet.to(cls.device)

        for itr in range(epoch * batches_per_epoch):
            optimizer.zero_grad()
            x, y = data_gen.__next__()
            x = x.view(-1, 28, 28, 1)
            x = torch.transpose(x, 1, 2)

            x = x.to(cls.device)
            y = y.to(cls.device)
            logits = cls.cnnNet(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            if itr % batches_per_epoch == 0:
                with torch.no_grad():
                    train_acc = cls.accuracy(cls.cnnNet, cls.train_loader)
                    val_acc = cls.accuracy(cls.cnnNet, cls.test_loader)
                    if val_acc > best_acc:
                        torch.save({"state_dict": cls.cnnNet.state_dict()}, "cnn2.pth")
                        best_acc = val_acc
                    logging.info(
                        "Epoch {:04d}"
                        "Train Acc {:.4f} | Test Acc {:.4f}".format(
                            itr // batches_per_epoch, train_acc, val_acc
                        )
                    )

                    print(
                        "Epoch {:04d}"
                        "Train Acc {:.4f} | Test Acc {:.4f}".format(
                            itr // batches_per_epoch, train_acc, val_acc
                        )
                    )


if __name__ == "__main__":
    args = parser.parse_args()

    trainObj = Train()
    trainObj.training()
