import os
import pickle
import numpy as np
from tqdm import tqdm
from time import time
import torch
from torch import nn
import torchmetrics
import matplotlib.pyplot as plt
from datasets import StrokeDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RNN(nn.Module):
    def __init__(self, h=256, num_layers=2, lstm_dropout=0.5, linear_dropout=0.1):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.h = h
        self.lstm = nn.LSTM(input_size=3, hidden_size=h, num_layers=num_layers, dropout=lstm_dropout)
        self.clf = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(h, 25),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out, _ = pad_packed_sequence(lstm_out)
        # out: L x batch x h
        # print(out[-1].shape)

        return self.clf(out[-1])


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


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


batch_size = 32
lr = 1e-4
weight_decay = 1e-3
epochs = 10
h = 256
num_layers = 2
patience = 5


def collate_fn(batch):
    y = torch.tensor([ins[1] for ins in batch], device=device)
    x = [torch.from_numpy(ins[0]).to(device) for ins in batch]
    x_lens = [len(x_) for x_ in x]
    x = pad_sequence(x)
    return x.to(device).float(), y, x_lens


def run():
    seed_everything(1)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    training_time = str(int(time() * 1000))
    save_folder = f"./model/{training_time}_lstm/"

    train_dataset = StrokeDataset(os.path.join('data/dataset_images', "train"))
    test_dataset = StrokeDataset(os.path.join('data/dataset_images', "test"))
    dev_dataset = StrokeDataset(os.path.join('data/dataset_images', "val"))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dev_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = RNN(h=h, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_epochs_loss = []
    valid_epochs_loss = []
    train_loss = []
    valid_loss = []

    for epoch in range(epochs):
        train_epoch_loss = []
        valid_epoch_loss = []

        model.train()
        for step, (x, y, x_lens) in enumerate(tqdm(train_dataloader)):
            # print(f'step: {step}, x.shape: {x.shape}, y: {y}')
            x_packed = pack_padded_sequence(x, x_lens, enforce_sorted=False)
            out = model(x_packed)

            loss = criterion(out, y)
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f'Epoch {epoch}, step {step} / {len(train_dataloader)}, loss: {np.mean(loss.item())}')
        train_epochs_loss.append(np.mean(train_epoch_loss))

        model.eval()
        with torch.no_grad():
            for step, (x, y, x_lens) in enumerate(tqdm(dev_dataloader)):
                x_packed = pack_padded_sequence(x, x_lens, enforce_sorted=False)
                out = model(x_packed)

                loss = criterion(out, y)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
            valid_epochs_loss.append(np.mean(valid_epoch_loss))

        # early stopping test
        early_stopping(valid_epochs_loss[-1], model=model,
                       path=os.path.join(save_folder, f"lstm_{epoch}.pth"))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "train_loss.npy"), train_loss)
    np.save(os.path.join(save_folder, "train_epoch_loss.npy"), train_epoch_loss)
    np.save(os.path.join(save_folder, "valid_epoch_loss.npy"), valid_epoch_loss)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss, '-o', label="train_loss")
    plt.plot(valid_epochs_loss, '-o', label="valid_loss")
    plt.title("validation_loss")
    plt.legend()
    plt.savefig(os.path.join(save_folder, f"loss_curve_lstm.png"), dpi=300)
    # plt.show()

    # test
    model.eval()
    test_acc = torchmetrics.Accuracy(num_classes=25)
    with torch.no_grad():
        for idx, (x, y, x_lens) in enumerate(tqdm(test_dataloader)):
            x_packed = pack_padded_sequence(x, x_lens, enforce_sorted=False)
            out = model(x_packed)
            test_acc(out.argmax(1), y)
        total_acc = test_acc.compute()
        print(f"test acc: {total_acc:.5f}")


if __name__ == '__main__':
    run()





