import argparse
import os
import yaml
import torch
import torchmetrics
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import FeatureDataset
from utils import device, seed_everything, EarlyStopping


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(50, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.clf(x)


if __name__ == "__main__":

    # hyper parameters
    num_epochs = 5
    batch_size = 512
    lr = 3e-4
    weight_decay = 1e-5
    patience = 20
    lr_adjust = {}  # epoch: lrate
    seed = 0

    seed_everything(seed)

    train_dataset = FeatureDataset('train.pickle')
    test_dataset = FeatureDataset('test.pickle')
    valid_dataset = FeatureDataset('valid.pickle')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = Model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    training_time = str(int(time() * 1000))
    save_folder = f"./model/{training_time}_clf/"

    for epoch in range(num_epochs):
        train_epoch_loss = []
        valid_epoch_loss = []

        # train
        model.train()
        for idx, (x, y) in enumerate(tqdm(train_dataloader)):
            y_pred = model(x.to(device))

            loss = criterion(y_pred, torch.squeeze(y).to(device))
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            '''
            if idx % (len(train_dataloader) // 1) == 0:
                print("epoch={}/{}, {}/{}of train, loss={}".format(
                    epoch, num_epochs, idx, len(train_dataloader), loss.item()))
            '''

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_epochs_loss.append(np.average(train_epoch_loss))

        # val
        model.eval()
        valid_acc = torchmetrics.Accuracy(25).to(device)
        train_acc = torchmetrics.Accuracy(25).to(device)
        with torch.no_grad():
            for idx, (x, y) in enumerate(tqdm(valid_dataloader)):
                y_pred = model(x.to(device))
                valid_acc(y_pred.argmax(1), torch.squeeze(y).to(device))
                #loss = criterion(y_pred, torch.squeeze(y).to(device))
                #valid_epoch_loss.append(loss.item())
                #valid_loss.append(loss.item())
            #valid_epochs_loss.append(np.average(valid_epoch_loss))
            valid_acc = valid_acc.compute()
            for idx, (x, y) in enumerate(tqdm(train_dataloader)):
                y_pred = model(x.to(device))
                train_acc(y_pred.argmax(1), torch.squeeze(y).to(device))
            train_acc = train_acc.compute()
            print(f"train acc: {train_acc:.4f}, valid acc: {valid_acc:.4f}")


    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "train_loss.npy"), train_loss)
    np.save(os.path.join(save_folder, "train_epochs_loss.npy"), train_epochs_loss)
    np.save(os.path.join(save_folder, "valid_epochs_loss.npy"), valid_epochs_loss)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss, '-o', label="train_loss")
    plt.plot(valid_epochs_loss, '-o', label="valid_loss")
    plt.title("validation_loss")
    plt.legend()
    plt.savefig(os.path.join(save_folder, f"loss_curve_clf.png"), dpi=300)
    # plt.show()

    # test
    model.eval()
    test_acc = torchmetrics.Accuracy(num_classes=25).to(device)
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(test_dataloader)):
            y_pred = model(x.to(device))
            test_acc(y_pred.argmax(1), torch.squeeze(y).to(device))
        total_acc = test_acc.compute()
        print(f"test acc: {total_acc:.5f}")


