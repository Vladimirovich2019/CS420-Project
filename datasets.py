import os
import cv2
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
            for img in imgs:
                self._data_path.append(os.path.join(folder, class_name, img))
                self._labels.append(class_id)
            self._len += len(imgs)
        
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        return np.transpose(cv2.imread(self._data_path[idx]), [2, 0, 1]), self._labels[idx]
    
    @property
    def num_classes(self): return len(self._classes)