import argparse
import os
import yaml
import torch
import torchmetrics
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as torchmodels
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from datasets import ImageStrokeDataset
from utils import device, collate_fn, seed_everything, EarlyStopping
from neuralline.rasterize import RasterIntensityFunc


class SeqEncoder(nn.Module):
    
    def __init__(self,
                 input_size,
                 hidden_size=512,
                 num_layers=2,
                 out_channels=1,
                 batch_first=False,
                 bidirect=True,
                 dropout=0,
                 requires_grad=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.batch_first = batch_first
        self.bidirect = bidirect
        self.proj_last_hidden = False

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=batch_first,
                           bidirectional=bidirect,
                           dropout=dropout)

        num_directs = 2 if bidirect else 1
        self.attend_fc = nn.Linear(hidden_size * num_directs, out_channels)

        if self.proj_last_hidden:
            self.last_hidden_size = hidden_size
            self.last_hidden_fc = nn.Linear(num_directs * num_layers * hidden_size, self.last_hidden_size)
        else:
            self.last_hidden_size = num_directs * num_layers * hidden_size
            self.last_hidden_fc = None

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, points, lengths):
        batch_size = points.shape[1]
        num_points = points.shape[0]
        point_dim = points.shape[2]


        if point_dim != self.input_size:
            points = points[:, :, :self.input_size]

        points_packed = pack_padded_sequence(points, lengths, batch_first=self.batch_first, enforce_sorted=False)
        hiddens_packed, (last_hidden, _) = self.rnn(points_packed) 

        intensities_act = torch.sigmoid(self.attend_fc(hiddens_packed.data))

        intensities_packed = PackedSequence(intensities_act, hiddens_packed.batch_sizes)
        intensities, _ = pad_packed_sequence(intensities_packed, batch_first=self.batch_first, total_length=num_points)

        # print(last_hidden.shape)
        last_hidden = last_hidden.view(batch_size, -1)

        if self.proj_last_hidden:
            last_hidden = F.relu(self.last_hidden_fc(last_hidden))

        return intensities, last_hidden


class SketchR2CNN(nn.Module):
    
    def __init__(self,
                 cnn_model,
                 rnn_input_size=3,
                 rnn_dropout=0.2,
                 img_size=28,
                 thickness=1.0,
                 num_categories=25,
                 intensity_channels=1,
                 train_cnn=True,
                 device=None):
        super().__init__()

        self.img_size = img_size
        self.thickness = thickness
        self.intensity_channels = intensity_channels
        self.eps = 1e-4
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        self.rnn = SeqEncoder(rnn_input_size, out_channels=intensity_channels, dropout=rnn_dropout)
        self.cnn: nn.Module = getattr(torchmodels, cnn_model)(pretrained=True)

        num_fc_in_features = self.cnn.fc.in_features
        self.fc = nn.Linear(num_fc_in_features, num_categories)
        self.cnn.fc = nn.Identity()

        nets.extend([self.rnn, self.cnn, self.fc])
        names.extend(['rnn', 'conv', 'fc'])
        train_flags.extend([True, train_cnn, True])

        self.to(device)

    def forward(self, points, points_offset, lengths):
        intensities, _ = self.rnn(points_offset, lengths)

        images = RasterIntensityFunc.apply(points, intensities, self.img_size, self.thickness, self.eps, self.device)
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        cnnfeat = self.cnn(images)
        logits = self.fc(cnnfeat)

        return logits, intensities, images


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
    #######################################################
    # train_cfg["cnn_model"] = 'resnet50'
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

    model = SketchR2CNN(train_cfg["cnn_model"], device=device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lrate, weight_decay=weight_decay)

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    training_time = str(int(time() * 1000))
    save_folder = f"./model/{training_time}_{train_cfg['cnn_model']}_r2cnn/"

    for epoch in range(num_epochs):
        train_epoch_loss = []
        valid_epoch_loss = []

        # train
        model.train()
        for idx, (x, x_stroke, y, stroke_len) in enumerate(tqdm(train_dataloader)):
            # stroke_packed = pack_padded_sequence(x_stroke, stroke_len, enforce_sorted=False)
            y_pred, _attention, _image = model(x, x_stroke, stroke_len)

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
                y_pred, _attention, _image = model(x, x_stroke, stroke_len)
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
            y_pred, _attention, _image = model(x, x_stroke, stroke_len)
            test_acc(y_pred.argmax(1), y.long().to(device))
        total_acc = test_acc.compute()
        print(f"test acc: {total_acc:.5f}")

