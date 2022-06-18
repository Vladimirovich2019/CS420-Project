import os
import torch
import random
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def collate_fn(batch):
    if len(batch[0]) == 3:
        images = torch.from_numpy(np.array([ins[0] for ins in batch])).float().to(device)
        strokes = [torch.from_numpy(np.concatenate([
            (ins[1][:, :2] - 0.6) / 20,
            ins[1][:, 2:]
        ], axis=1)).to(device) for ins in batch]
        y = torch.tensor([ins[2] for ins in batch], device=device)

        strokes_lens = [len(stroke) for stroke in strokes]
        strokes = pad_sequence(strokes)
        return images, strokes.to(device).float(), y, strokes_lens
    
    strokes = [torch.from_numpy(np.concatenate([
        (ins[0][:, :2] - 0.6) / 20,
        ins[0][:, 2:]
    ], axis=1)).to(device) for ins in batch]

    y = torch.tensor([ins[1] for ins in batch], device=device)

    strokes_lens = [len(stroke) for stroke in strokes]
    strokes = pad_sequence(strokes)
    return strokes.to(device).float(), y, strokes_lens
    
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