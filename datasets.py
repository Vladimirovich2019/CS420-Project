import os
import cv2
import pickle
import numpy as np
import torch.utils.data as data


class ImageDataset(data.Dataset):
    def __init__(self, folder:str):
        self._classes = sorted(os.listdir(folder))
        self._data_path = []
        self._labels = []
        self._len = 0
        for class_id, class_name in enumerate(self._classes):
            imgs = os.listdir(os.path.join(folder, class_name))
            imgs.sort(key=lambda s: int(s[:-4]))
            for img in imgs:
                if not img.endswith('png'): continue
                self._data_path.append(os.path.join(folder, class_name, img))
                self._labels.append(class_id)
            self._len += len(imgs)
        
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        return np.transpose(cv2.imread(self._data_path[idx]), [2, 0, 1]), self._labels[idx]
    
    @property
    def num_classes(self): return len(self._classes)


class StrokeDataset(data.Dataset):
    def __init__(self, folder: str):
        self._classes = sorted(os.listdir(folder))
        self._strokes = []
        self._labels = []
        self._len = 0
        for class_id, class_name in enumerate(self._classes):
            with open(os.path.join(folder, class_name, 'strokes.pickle'), 'rb') as f:
                strokes = pickle.load(f)
                self._strokes.extend(list(strokes))
            self._labels.extend([class_id]*strokes.shape[0])
            self._len += strokes.shape[0]
        self._strokes = np.array(self._strokes)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self._strokes[idx], self._labels[idx]

    @property
    def num_classes(self):
        return len(self._classes)

class StrokeDatasetOrig(data.Dataset):
    def __init__(self, folder="./data/dataset", mode:str="train"):
        self._classes = sorted(os.listdir(folder))
        self._strokes = []
        self._labels = []
        self._len = 0
        for class_id, class_name in enumerate(self._classes):
            strokes = np.load(os.path.join(folder, class_name), allow_pickle=True, encoding="bytes")[mode].tolist()[:10]
            self._strokes.extend(strokes)
            self._labels.extend([class_id] * len(strokes))
            self._len += len(strokes)
        self._strokes = np.array(self._strokes, dtype=object)
    
    def __len__(self): return self._len
    
    def __getitem__(self, idx): return self._strokes[idx], self._labels[idx]
    
    @property
    def num_classes(self): return len(self._classes)