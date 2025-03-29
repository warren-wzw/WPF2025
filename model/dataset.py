from calendar import month
import os
import torch
import tqdm
import json
import random
import math
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime

from PIL import Image
from torch.utils.data import  Dataset
"""Dataset"""   
def min_max_normalize(image):
    np_image = np.array(image).astype(np.float32)
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    return torch.tensor(np_image)

def visual_result(input,filename):
    if len(input.shape)==4:
        np_image = input[0].cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    elif len(input.shape)==3:
        np_image = input.cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    if np_image.min()<0:    
        np_image = np_image * 0.5 + 0.5  # 假设图像已归一化为[-1, 1]
    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(filename)  # 在绘制图像后保存  
    
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image=transform(image)
    #visual_result(image,"out.jpg")
    return image
    
def PreprocessCacheData(data_path,label_path,cache_file,cache=False,shuffle=True):
    if cache ==True and cache_file is not None and os.path.exists(cache_file):
        print("Loading features from cached file ", cache_file)
        features = torch.load(cache_file)
    else:
        print("Creating features from dataset at ", data_path)
        def processdata(data_path):
            print()
            return 0
        
        data=processdata(data_path)
        images,labels = [],[]
        for i  in range(len(data)):
            image = data[i]
            label = data[i]
            images.append(image)
            labels.append(label)    
        features=[]
        total_iterations = len(images) 
        for image,label in tqdm.tqdm(zip(images,labels),total=total_iterations):
            processed_image=preprocess_image(image)
            feature={
                "images":processed_image,
                "label":label
            }
            features.append(feature)
 
        if shuffle:
            random.shuffle(features)
        if cache==True and not os.path.exists(cache_file):
            print("Saving features into cached file ", cache_file)
            torch.save(features, cache_file)
    return features

class CacheDataset(Dataset):
    def __init__(self,features,num_instances):
        self.feature=features
        self.num_instances=num_instances
    
    def __len__(self):
        return int(self.num_instances)
    
    def __getitem__(self, index):
        feature = self.feature[index]
        image=feature["images"]
        label=feature["label"]
        return image,label
    
class OnlineCacheDataset_pack(Dataset):   #only when json is standard json form,it will speed up
    def __init__(self, root_dir: str, shuffle=False,transform=None):
        data_dir = root_dir
        assert os.path.exists(data_dir), f"data directory '{data_dir}' not found."
        with open(data_dir, 'r') as file:
            data = json.load(file) 
        self.datas=data
        self.total_num = len(data)
        self.feature_names = ["ghi", "poai", "sp", "t2m", "tcc", "tp", "u100", "v100"]
        self.num=1
    
    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        label_wind = []  # 存放最终的标签
        label_light = []  # 存放最终的标签
        data_list = []  # 存放所有的数据
        num = self.num  # T 个时间步
        def replace_nan_with_zero(value):
            return 0 if math.isnan(value) else value
        label_wind.extend([
            replace_nan_with_zero(self.datas[item]["power0"]),
            replace_nan_with_zero(self.datas[item ]["power1"]),
            replace_nan_with_zero(self.datas[item ]["power2"]),
            replace_nan_with_zero(self.datas[item ]["power3"]),
            replace_nan_with_zero(self.datas[item ]["power4"])
        ])

        label_light.extend([
            replace_nan_with_zero(self.datas[item ]["power5"]),
            replace_nan_with_zero(self.datas[item ]["power6"]),
            replace_nan_with_zero(self.datas[item]["power7"]),
            replace_nan_with_zero(self.datas[item]["power8"]),
            replace_nan_with_zero(self.datas[item ]["power9"])
        ])
        
        for i in range(num):
            timestamp = [datetime.strptime(self.datas[item + i]["TimeStamp"][0], "%Y-%m-%d %H:%M:%S").timestamp()]
            features=[]
            for j in range(10):
                label_history=self.datas[item + i][f"power{j}"]
                feature = [self.datas[item + i][f"{name}_{j}"] for name in self.feature_names ]
                features=features+feature
            step_data = features
            data_list.append(step_data)
        
        return label_wind,label_light,timestamp, data_list  # 返回标签和所有数据


    @staticmethod
    def collate_fn(batch):
        label_wind,label_light,timestamp,data_list = tuple(zip(*batch))
        label_wind = torch.as_tensor(label_wind) 
        label_light = torch.as_tensor(label_light)    # 转换标签为张量
        timestamp = torch.as_tensor(label_light) 
        data_list = torch.as_tensor(data_list)    # 转换标签为张量
       
        return label_wind,label_light,timestamp,data_list

def replace_nan_with_zero(value):
            return 0 if math.isnan(value) else value
        
class OnlineCacheDataset(Dataset):   #only when json is standard json form,it will speed up
    def __init__(self, root_dir: str, shuffle=False,transform=None):
        label = []  # 存放最终的标签
        data_list = []  # 存放所有的数据
        station=[]
        month=[]
        day=[]
        hour=[]
        minute=[]
        self.num=1
        data_dir = root_dir
        with open(data_dir, 'r') as file:
            data = json.load(file) 
        self.datas=data
        self.feature_names = ["ghi", "poai", "sp", "t2m", "tcc", "tp", "u100", "v100"]
        for item in range(len(data)):
            timestamp = datetime.strptime(self.datas[item]["TimeStamp"][0], "%Y-%m-%d %H:%M:%S")
            for i in range(10):
                label.extend([replace_nan_with_zero(self.datas[item][f"power{i}"])])
                station.extend([i])
                month.extend([timestamp.month])
                day.extend([timestamp.day])
                hour.extend([timestamp.hour])
                minute.extend([timestamp.minute])
                feature = [self.datas[item][f"{name}_{i}"] for name in self.feature_names]
                data_list.append(feature)
        self.label=label
        self.station=station
        self.month=month
        self.day=day
        self.hour=hour
        self.minute=minute
        self.data_list=data_list
        self.total_num = len(self.label)
    
    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        label = []  # 存放最终的标签
        data_list = []  # 存放所有的数据
        station=[]
        month=[]
        day=[]
        hour=[]
        minute=[]
        label.extend([self.label[item]])
        station.extend([self.station[item]])
        month.extend([self.month[item]])
        day.extend([self.day[item]])
        hour.extend([self.hour[item]])
        minute.extend([self.minute[item]])
        data_list.extend([self.data_list[item]])

        return label,station,month,day,hour,minute,data_list  # 返回标签和所有数据


    @staticmethod
    def collate_fn(batch):
        label,station,month,day,hour,minute,data_list = tuple(zip(*batch))
        label = torch.as_tensor(label) 
        station = torch.as_tensor(station)    # 转换标签为张量
        month = torch.as_tensor(month) 
        day = torch.as_tensor(day) 
        hour = torch.as_tensor(hour) 
        minute = torch.as_tensor(minute) 
        data_list = torch.as_tensor(data_list)    # 转换标签为张量
       
        return label,station,month,day,hour,minute,data_list