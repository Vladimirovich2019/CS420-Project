import argparse
import os
import yaml
import torch
import torchmetrics
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as torchmodels
from time import time
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datasets import ImageStrokeDataset
from utils import device, collate_fn, seed_everything, EarlyStopping


class RNN(nn.Module):
    def __init__(self, h=256, num_layers=2, lstm_dropout=0.5, linear_dropout=0.1):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.h = h
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5, padding=2),
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5, padding=2),
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        )
        self.lstm = nn.LSTM(input_size=3, hidden_size=h, num_layers=num_layers, dropout=lstm_dropout)
        self.clf = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(h, 25),
            nn.Sigmoid()
        )

    def forward(self, x, x_lens):
        # x: num_points x batch x 3
        conv_out = self.conv(x.reshape((x.shape[1], 3, x.shape[0])))
        # conv_out: batch x 3 x num_points
        x_packed = pack_padded_sequence(conv_out.reshape((conv_out.shape[2], conv_out.shape[0], 3)),
                                        x_lens, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(x_packed)
        out, _ = pad_packed_sequence(lstm_out)
        # out: L x batch x h
        # print(out.mean(dim=0).shape)

        return self.clf(out.mean(dim=0))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = RNN()
        self.cnn: nn.Module = getattr(torchmodels, 'efficientnet_b7')(pretrained=True)
        self.cnn_h = self.cnn.classifier[-1].in_features
        self.cnn.classifier[-1] = nn.Linear(self.cnn_h, 25)
        self.rnn.load_state_dict(torch.load('model/1655552467390_lstm/lstm_6.pth'))
        self.cnn.load_state_dict(torch.load('model/1655450563334_efficientnet_b7/efficientnet_b7_13.pth'))
        self.cnn.classifier[-1] = nn.Identity()
        self.rnn.clf[-1] = nn.Identity()
        self.rnn.clf[-2] = nn.Identity()
        for p in self.cnn.parameters(): p.requires_grad = False
        for p in self.rnn.parameters(): p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(256 + self.cnn_h, 256),
            nn.ReLU(),
            nn.Linear(256, 25),
            nn.Sigmoid()
        )
        for p in self.classifier.parameters(): p.requires_grad = True

    def forward(self, image, stroke, stroke_lens):
        cnn_out = self.cnn(image)
        rnn_out = self.rnn(stroke, stroke_lens)

        feature = torch.concat([cnn_out, rnn_out], dim=1)

        return self.classifier(feature)


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
    lr_rnn = 3e-3
    weight_decay_rnn = 1e-4
    #######################################################
    train_cfg["cnn_model"] = 'efficientnet_b7'
    #######################################################

    seed_everything(seed)

    train_dataset = ImageStrokeDataset(os.path.join(train_cfg["data"], "train"))
    test_dataset = ImageStrokeDataset(os.path.join(train_cfg["data"], "test"))
    valid_dataset = ImageStrokeDataset(os.path.join(train_cfg["data"], "val"))

    class_num_train = train_dataset.num_classes
    class_num_test = test_dataset.num_classes
    if class_num_test != class_num_train: raise ValueError("unmatched train and test class numbers")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = Model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam([
        {'params': model.classifier.parameters(), 'lr': lrate, 'weight_decay': weight_decay},
    ])

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    training_time = str(int(time() * 1000))
    save_folder = f"./model/{training_time}_{train_cfg['cnn_model']}_lstm/"

    for epoch in range(num_epochs):
        train_epoch_loss = []
        valid_epoch_loss = []

        # train
        model.train()
        for idx, (x, x_stroke, y, stroke_len) in enumerate(tqdm(train_dataloader)):
            y_pred = model(x, x_stroke, stroke_len)

            loss = criterion(y_pred, y.long())
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader) // 2) == 0:
                print("epoch={}/{}, {}/{}of train, loss={}".format(
                    epoch, num_epochs, idx, len(train_dataloader), loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_epochs_loss.append(np.average(train_epoch_loss))

        # val
        model.eval()
        with torch.no_grad():
            for idx, (x, x_stroke, y, stroke_len) in enumerate(tqdm(valid_dataloader)):
                y_pred = model(x, x_stroke, stroke_len)
                loss = criterion(y_pred, y.long())
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
    np.save(os.path.join(save_folder, "train_epochs_loss.npy"), train_epochs_loss)
    np.save(os.path.join(save_folder, "valid_epochs_loss.npy"), valid_epochs_loss)

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
    test_acc = torchmetrics.Accuracy(num_classes=class_num_train).to(device)
    with torch.no_grad():
        for idx, (x, x_stroke, y, stroke_len) in enumerate(tqdm(test_dataloader)):
            y_pred = model(x, x_stroke, stroke_len)
            test_acc(y_pred.argmax(1), y.long())
        total_acc = test_acc.compute()
        print(f"test acc: {total_acc:.5f}")


