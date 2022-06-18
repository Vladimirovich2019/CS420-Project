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
            imgs.remove('strokes.pickle')
            imgs.sort(key=lambda s: int(s[:-4]))
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


class ImageStrokeDataset(data.Dataset):
    def __init__(self, folder: str):
        self._classes = sorted(os.listdir(folder))
        self._data_path = []
        self._strokes = []
        self._labels = []
        self._len = 0

        for class_id, class_name in enumerate(self._classes):
            with open(os.path.join(folder, class_name, 'strokes.pickle'), 'rb') as f:
                strokes = pickle.load(f)
                self._strokes.extend(list(strokes))
            self._labels.extend([class_id]*strokes.shape[0])
            self._len += strokes.shape[0]

            imgs = os.listdir(os.path.join(folder, class_name))
            imgs.remove('strokes.pickle')
            imgs.sort(key=lambda s: int(s[:-4]))
            for img in imgs:
                self._data_path.append(os.path.join(folder, class_name, img))
        self._strokes = np.array(self._strokes)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return np.transpose(cv2.imread(self._data_path[idx]), [2, 0, 1]), self._strokes[idx], self._labels[idx]

    @property
    def num_classes(self):
        return len(self._classes)


# for toy code
class StrokeDatasetOrig(data.Dataset):
    def __init__(self, folder="./data/dataset", mode:str="train"):
        self._classes = sorted(os.listdir(folder))
        self._strokes = []
        self._labels = []
        self._len = 0
        for class_id, class_name in enumerate(self._classes):
            strokes = np.load(os.path.join(folder, class_name), allow_pickle=True, encoding="bytes")[mode].tolist()
            self._strokes.extend(strokes)
            self._labels.extend([class_id] * len(strokes))
            self._len += len(strokes)
        self._strokes = [np.array(stroke, dtype=np.float32) / 256 for stroke in self._strokes]
        self._strokes = np.array(self._strokes, dtype=object)
    
    def __len__(self): return self._len
    
    def __getitem__(self, idx): return self._strokes[idx], self._labels[idx]
    
    @property
    def num_classes(self): return len(self._classes)


#### for robustness study
def flip_img_h(transposed_img:np.array): return np.flip(transposed_img, axis=1).copy()
def flip_img_w(transposed_img:np.array): return np.flip(transposed_img, axis=2).copy()
def flip_img(transposed_img:np.array): return np.flip(transposed_img).copy()
def flip_strokes_h(strokes:np.array): raise NotImplementedError


class ImageDatasetAug(ImageDataset):
    def __init__(self, folder: str):
        super().__init__(folder)
        # change the preprocessor here
        self.processor = flip_img_w
    
    def __getitem__(self, idx):
        return self.processor(np.transpose(cv2.imread(self._data_path[idx]), [2, 0, 1])), self._labels[idx]


class StrokeDatasetAug(StrokeDataset):
    def __init__(self, folder: str):
        super().__init__(folder)


def transpose_back(transposed_img:np.array): return np.transpose(transposed_img, [1, 2, 0])


def test_preproc():
    img_path = "./data/dataset_images/test/cow/3.png"
    transposed_img = np.transpose(cv2.imread(img_path), [2, 0, 1])
    img_flip_h = flip_img_h(transposed_img)
    img_flip_w = flip_img_w(transposed_img)
    
    cv2.imshow("Original", transpose_back(transposed_img))
    cv2.imshow("Flipped H", transpose_back(img_flip_h))
    cv2.imshow("Flipped W", transpose_back(img_flip_w))
    cv2.waitKey(-1)


def test_stroke():
    dataset = StrokeDatasetOrig()
    stroke, label = dataset[0]
    print(stroke.shape)
    print(stroke)


if __name__ == "__main__":
    test_preproc()
    
    test_stroke()