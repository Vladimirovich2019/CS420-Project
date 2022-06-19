import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as torchmodels
from tqdm import tqdm
from datasets import noise_img, transpose_back
from utils import seed_everything


def feat_vis(outputs, name:str):
    print(name)
    
    plt.figure(figsize=(30, 30))
    
    layer_viz = outputs[0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis('off')
    plt.savefig(f'./img/{name}_noised.png')
    
    plt.clf()
    
    layer_viz = outputs[1, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis('off')
    plt.savefig(f'./img/{name}_orig.png')
    
    plt.clf()

if __name__ == "__main__":
    class_names = sorted([class_name.replace("sketchrnn_", "").replace(".npz", "") for class_name in os.listdir("./data/dataset")])
    
    img_path = "./data/dataset_images/test/dog/890.png"
    img = np.transpose(cv2.imread(img_path), [2, 0, 1])
    
    img_noised = noise_img(img)
    cv2.imwrite("./img/noise.png", transpose_back(img_noised))
    input = torch.from_numpy(np.array([img_noised, img], dtype=np.float32))
    
    with torch.no_grad():
        model = torchmodels.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 25)
        model_path = "./model/trained/resnet50_9.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        # output = model.relu(model.bn1(model.conv1(input)))
        output = model.conv1(input)
        idxs = torch.argmax(model(input), dim=1).numpy()
        print(class_names[idxs[0]], class_names[idxs[1]])
        feat_vis(output, "resnet50")
        
        model = torchmodels.efficientnet_b7()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 25)
        model_path = "./model/trained/efficientnet_b7_13.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        output = model.features[0][0](input)
        idxs = torch.argmax(model(input), dim=1).numpy()
        print(class_names[idxs[0]], class_names[idxs[1]])
        feat_vis(output, "efficientnet_b7")
        
        model = torchmodels.mobilenet_v2()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 25)
        model_path = "./model/trained/mobilenet_v2_9.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        output = model.features[0][0](input)
        idxs = torch.argmax(model(input), dim=1).numpy()
        print(class_names[idxs[0]], class_names[idxs[1]])
        feat_vis(output, "mobilenet_v2")
    
    
    