import argparse
import os
import yaml
import torch
import random
import torchmetrics
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as torchmodels
from time import time
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import ImageDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="./configs/default.yaml")
    ARGS = parser.parse_args()

    try:
        with open(ARGS.config, "r") as f:
            train_cfg = yaml.load(f, Loader=yaml.Loader)
    except Exception as e:
        print(e)
        exit(-1)

    # hyper parameters
    num_epochs = train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    lrate = train_cfg["lrate"]
    weight_decay = train_cfg["weight_decay"]
    patience = train_cfg["patience"]
    lr_adjust = {}  # epoch: lrate
    plot_loss_curve = train_cfg["plot_loss_curve"]
    seed = train_cfg["seed"]

    seed_everything(seed)

    train_dataset = ImageDataset(os.path.join(train_cfg["data"], "train"))
    test_dataset = ImageDataset(os.path.join(train_cfg["data"], "test"))
    valid_dataset = ImageDataset(os.path.join(train_cfg["data"], "val"))

    class_num_train = train_dataset.num_classes
    class_num_test = test_dataset.num_classes
    if class_num_test != class_num_train: raise ValueError("unmatched train and test class numbers")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model: nn.Module = getattr(torchmodels, train_cfg["cnn_model"])(num_classes=class_num_train)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lrate, weight_decay=weight_decay)

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    training_time = str(int(time() * 1000))
    save_folder = f"./model/{training_time}_{train_cfg['cnn_model']}/"

    for epoch in range(num_epochs):
        train_epoch_loss = []
        valid_epoch_loss = []

        # train
        model.train()
        for idx, (x, y) in enumerate(tqdm(train_dataloader)):
            y_pred = model(x.to(torch.float32).to(device))
            optimizer.zero_grad()
            loss = criterion(y_pred, y.long().to(device))
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader) // 2) == 0:
                print("epoch={}/{}, {}/{}of train, loss={}".format(
                    epoch, num_epochs, idx, len(train_dataloader), loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))

        # val
        model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(tqdm(valid_dataloader)):
                y_pred = model(x.to(torch.float32).to(device))
                loss = criterion(y_pred, y.long().to(device))
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
            valid_epochs_loss.append(np.average(valid_epoch_loss))

        # early stopping test
        early_stopping(valid_epochs_loss[-1], model=model,
                       path=os.path.join(save_folder, f"{train_cfg['cnn_model']}_{epoch}.pth"))
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # adjust learning rate
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "train_loss.npy"), train_loss)
    np.save(os.path.join(save_folder, "train_epoch_loss.npy"), train_epoch_loss)
    np.save(os.path.join(save_folder, "valid_epoch_loss.npy"), valid_epoch_loss)

    if plot_loss_curve:
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(train_loss[:])
        plt.title("train_loss")
        plt.subplot(122)
        plt.plot(train_epochs_loss, '-o', label="train_loss")
        plt.plot(valid_epochs_loss, '-o', label="valid_loss")
        plt.title("validation_loss")
        plt.legend()
        plt.savefig(os.path.join(save_folder, f"loss_curve_{train_cfg['cnn_model']}.png"), dpi=300)
        # plt.show()

    # test
    model.eval()
    test_acc = torchmetrics.Accuracy(num_classes=class_num_train)
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(test_dataloader)):
            y_pred = model(x.to(torch.float32).to(device))
            test_acc(y_pred.argmax(1), y.long().to(device))
        total_acc = test_acc.compute()
        print(f"test acc: {total_acc:.5f}")

