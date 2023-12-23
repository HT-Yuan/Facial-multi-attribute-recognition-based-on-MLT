from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.nn import functional as F
import time
from sklearn.metrics import precision_recall_fscore_support as score
from efficientnet.model import EfficientNet
import os
from PIL import Image

# some parameters
use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_dir = 'Predict'
batch_size = 16
lr = 0.01
momentum = 0.9
num_epochs = 60
input_size = 224

net_name = 'efficientnet-b0'



# 数据集的预处理，以及数据载入（分批次）
def loaddata(data_dir):
    data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    image_datasets = datasets.ImageFolder(data_dir, data_transforms)
    #class_to_idx
    
    # # num_workers=0 if CPU else =1
    # dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
    #                                                   batch_size=batch_size,
    #                                                   shuffle=shuffle, num_workers=1) for x in [set_name]}

    data_set_sizes = len(image_datasets)
    # print(data_set_sizes)
    image_datasets = torch.utils.data.DataLoader(image_datasets)
    return image_datasets






def test_model(model):
    model.eval()
    data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # 读取每批次数据

    
    inputs = Image.open('Predict/20231027145607.jpg')
    # inputs = Image.open('Predict/39_0_0_20170117165645001.jpg')
    inputs = data_transforms(inputs)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = Variable(inputs.cuda())

    # UTK_Face


    # output_age,output_gender,output_race,pro_age = model_ft(inputs)
        
        

    # pro_age = pro_age > 0.5
    # preds_age = torch.sum(pro_age, dim=1)
    
    # _, preds_gender = torch.max(output_gender.data, 1)
    # _, preds_race = torch.max(output_race.data, 1)

    # print('age:',preds_age.tolist(), 'gender:',preds_gender.tolist(), 'race:',preds_race.tolist())



    # Adience

    output_age, output_gender ,pro_age= model_ft(inputs)
  
   

    # st = time.time()
    # i = 0
    # while i < 10:
    #     # output_age = model_ft(inputs)
    #     output_age, output_gender = model_ft(inputs)
    #     i += 1
    
    
    # et = time.time()

    # 将输出值转为预测值
    pro_age = pro_age > 0.5
    _, preds_age = torch.max(output_age.data, 1)
    preds_age = torch.sum(pro_age, dim=1)
    _, preds_gender = torch.max(output_gender.data, 1)
    print('age:',preds_age.tolist(), 'gender:',preds_gender.tolist())


    
    # print(preds_age.tolist(),(et-st)*1000/10)

    
    





if __name__ == "__main__":




    # model_ft = torch.load('mydata/model/efficientnet-b0(b0_data).pth') #可单独使用测试函数

    # model_ft = torch.load('UTK_Face/model/efficientnet-b0(utk_mlt_e).pth') 
    model_ft = torch.load('mydata/model/efficientnet-b0(mlt_adience_EEE).pth') 
    total_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(total_params)


    test_model(model_ft)
