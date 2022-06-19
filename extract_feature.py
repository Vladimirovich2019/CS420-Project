import argparse
import os
import pickle

import yaml
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as torchmodels
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datasets import ImageStrokeDataset
from utils import device, collate_fn, seed_everything


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
        for p in self.cnn.parameters(): p.requires_grad = False
        for p in self.rnn.parameters(): p.requires_grad = False

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
    batch_size = train_cfg["batch_size"]
    seed = train_cfg["seed"]

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
    model.eval()
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    valid_x = []
    valid_y = []

    # train
    for idx, (x, x_stroke, y, stroke_len) in enumerate(tqdm(train_dataloader)):
        feature = model(x, x_stroke, stroke_len)
        train_x.append(feature)
        train_y.append(y.reshape((-1, 1)))

    # val
    for idx, (x, x_stroke, y, stroke_len) in enumerate(tqdm(valid_dataloader)):
        feature = model(x, x_stroke, stroke_len)
        valid_x.append(feature)
        valid_y.append(y.reshape((-1, 1)))

    # test
    for idx, (x, x_stroke, y, stroke_len) in enumerate(tqdm(test_dataloader)):
        feature = model(x, x_stroke, stroke_len)
        test_x.append(feature)
        test_y.append(y.reshape((-1, 1)))

    for lst in [train_x, train_y, test_x, test_y, valid_x, valid_y]:
        lst = torch.concat(lst)

    with open('train.pickle', 'wb') as f: pickle.dump([train_x, train_y], f)
    with open('valid.pickle', 'wb') as f: pickle.dump([valid_x, valid_y], f)
    with open('test.pickle', 'wb') as f: pickle.dump([test_x, test_y], f)
