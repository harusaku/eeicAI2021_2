import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
class MyDataset(Dataset):
    def __init__(self, category_ls, root=None, train=True, transform=None):
        self.root = root
        self.transform = transform
        mode = 'train' if train else 'test'

        self.all_data = []
        for category_word in category_ls:
            self.category = category_word
            # 画像パスとそのラベルのセットをself.all_dataに入れる
            data = np.load(self.root+ self.category + ".npy")
            # self.data = [[data_i.reshape(28,28),self.category] for data_i in data]
            for data_i in data:
                self.all_data.append([data_i.reshape(28,28),self.category])

    def __len__(self):
        #データセットの数を返す関数
        return len(self.all_data)
        
    def __getitem__(self, idx):
        # TODO 画像とラベルの読み取り
        #self.all_dataを用いる
        data_img = Image.fromarray(self.all_data[idx][0]).convert("L")
        if self.transform is not None:
            data_img = self.transform(data_img)
        label = self.all_data[idx][1]
        return [data_img, label]