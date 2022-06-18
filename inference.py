import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.models as torchmodels
from datasets import flip_img_h, flip_img_w

k = 5

def argmax_topn(a:np.array):
    ind = np.argpartition(a, -k)[-k:]
    return ind[np.argsort(a[ind])][::-1]

if __name__ == "__main__":    
    # load class names
    class_names = sorted([class_name.replace("sketchrnn_", "").replace(".npz", "") for class_name in os.listdir("./data/dataset")])
    
    model = torchmodels.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 25)
    model_path = "./model/resnet/resnet50_9.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # print(model)
    
    model.eval()
    test_folder = "./data/dataset_images/test/dog"
    softmax = nn.Softmax(dim=1)
    img_filenames = (os.listdir(test_folder))
    random.shuffle(img_filenames)
    img_filenames = ["199.png"]
    with torch.no_grad():
        for img_filename in img_filenames:
            img = cv2.imread(os.path.join(test_folder, img_filename))
            print(img_filename)
            cv2.imshow(img_filename, img)
            cv2.waitKey(-1)
            img = np.transpose(img, [2, 0, 1])
            img_flip_w = flip_img_w(img)
            img_flip_h = flip_img_h(img)
            
            img = torch.from_numpy(np.array([img])).float()
            img_flip_w = torch.from_numpy(np.array([img_flip_w])).float()
            img_flip_h = torch.from_numpy(np.array([img_flip_h])).float()
            score = softmax(model(img)).numpy()[0]
            score_w = softmax(model(img_flip_w)).numpy()[0]
            score_h = softmax(model(img_flip_h)).numpy()[0]
            
            ind = argmax_topn(score)
            ind_w = argmax_topn(score_w)
            ind_h = argmax_topn(score_h)
            
            for idx in ind: print(class_names[idx], f"{score[idx] * 100: .2f}%")
            print("==========")
            for idx in ind_w: print(class_names[idx], f"{score_w[idx] * 100: .2f}%")
            print("==========")
            for idx in ind_h: print(class_names[idx], f"{score_h[idx] * 100: .2f}%")
            exit()
    