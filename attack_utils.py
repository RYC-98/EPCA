import torch
import random
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
from foolbox import PyTorchModel
import torchvision.models as models
from datetime import datetime
import pandas as pd

def get_model(args,device):
    model_name = args.model_name
    if model_name == 'resnet-18':
        model = models.resnet18(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.to(device)
            std = std.to(device)

        preprocessing = dict(mean=mean, std=std, axis=-3) # 构造3个字典
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'inception-v3':
        model = models.inception_v3(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])  # 这个有待商榷
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'vgg-16':
        model = models.vgg16(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'resnet-50':
        model = models.resnet50(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'resnet-101':
        model = models.resnet101(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'densenet-121':
        model = models.densenet121(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel


def get_label(logit):
    _, predict = torch.max(logit, 1) # 逐行; 返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引。
    return predict



def save_results(args,my_intermediates, n):                                 # n = len(images)
    path = args.output_folder
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path):
        os.mkdir(path)
    numpy_results = np.full((n * 3, args.max_queries), np.nan)              # 前面是shape; 后面是填充的内容 
    for i, my_intermediate in enumerate(my_intermediates):
        length = len(my_intermediate)
        for j in range(length):
            numpy_results[3 * i][j] = my_intermediate[j][0]                 # 一个样本站三行, j代表第几次调用核心函数 get_x_hat_in_2d             
            numpy_results[3 * i + 1][j] = my_intermediate[j][1]
            numpy_results[3 * i + 2][j] = my_intermediate[j][2]
    pandas_results = pd.DataFrame(numpy_results)
    # pandas_results.to_csv(os.path.join(path,'results.csv'))
    pandas_results.to_csv(os.path.join(path,args.results_name))    
    print('save results to:{}'.format(os.path.join(path,args.results_name)))


def read_imagenet_data_specify(args, device):
    images = []
    labels = []
    info = pd.read_csv(args.csv)
    selected_image_paths = []
    for d_i in range(len(info)):                                        # 长度是图片总数
        image_path = info.iloc[d_i]['ImageName']                        # 每一行的图片路径
        image = Image.open(os.path.join(args.dataset_path,image_path))  # PIL包的图片读取
        image = image.convert('RGB')
        image = image.resize((args.side_length, args.side_length))      # v3: 299*299
        image = np.asarray(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))                          # 读取的格式(h,w,c) 转换成torch格式 (c,h,w) 
        groundtruth = info.iloc[d_i]['Label']                           # 每一行的图片的标签
        images.append(image)                         # images = []            
        labels.append(groundtruth)                   # labels = []   
        selected_image_paths.append(image_path)      # selected_image_paths = []
    images = np.stack(images)                        # 从list变成四维数组 [b,c,h,w]
    labels = np.array(labels)
    images = images / 255                            # 变到 0-1 区间 
    images = torch.from_numpy(images).to(device)
    labels = torch.from_numpy(labels).to(device).long() # label 64 整型
    return images, labels, selected_image_paths      # 读进来后就是 torch 数据


