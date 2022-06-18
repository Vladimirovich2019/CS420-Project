import os
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as torchmodels
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def plot_conv(writer, model):
    for name, param in tqdm(model.named_parameters()):
        if 'conv' in name and 'weight' in name:
            in_channels = param.size()[1]	# 输入通道
            out_channels = param.size()[0]   # 输出通道
            k_w, k_h = param.size()[3], param.size()[2]   # 卷积核的尺寸
            kernel_all = param.view(-1, 1, k_w, k_h)  # 每个通道的卷积核
            print(kernel_all.shape, in_channels)
            kernel_grid = torchvision.utils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels * 4)
            writer.add_image(f'{name}_all', kernel_grid, global_step=0) 
            exit()
            
if __name__ == "__main__":
    class_names = sorted([class_name.replace("sketchrnn_", "").replace(".npz", "") for class_name in os.listdir("./data/dataset")])
    model = torchmodels.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 25)
    model_path = "./model/resnet/resnet50_9.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    writer = SummaryWriter()
    
    plot_conv(writer, model)