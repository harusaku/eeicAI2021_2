import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import clip
import random
class MyDataset(Dataset):
    def __init__(self, category_ls, root=None, train=True, transform=None, device='cpu'):
        self.root = root
        self.transform = transform
        mode = 'train' if train else 'test'
        
        # Load CLIP info before training
        # to avoid moving data between CPU and GPU
        ### CLIP Features
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        # TRICK5: trying to improve the behaviour of CLIP by identifying the subject as sketches
        # for word in category_ls:
        #     word = "a sketch of " + word
        tokenized_text = clip.tokenize(category_ls).to(device)
        with torch.no_grad():
            self.text_features = clip_model.encode_text(tokenized_text)

        self.all_data = []
        for i in range(len(category_ls)):
            self.category = category_ls[i]
            self.text_feature = self.text_features[i].float()
            # 画像パスとそのラベルのセットをself.all_dataに入れる
            data = np.load(self.root+ self.category + ".npy")
            # self.data = [[data_i.reshape(28,28),self.category] for data_i in data]
            for data_i in data:
                self.all_data.append([data_i.reshape(28,28),self.category, self.text_feature])

    def __len__(self):
        #データセットの数を返す関数
        return len(self.all_data)
        
    def __getitem__(self, idx):
        # TODO 画像とラベルの読み取り
        #self.all_dataを用いる
        data_img = Image.fromarray(self.all_data[idx][0]).convert("L")
        if self.transform is not None:
            data_img = self.transform(data_img)
        category = self.all_data[idx][1]
        text_feature = self.all_data[idx][2]
        rand_img = Image.fromarray(random.choice(self.all_data)[0]).convert("L")
        if self.transform is not None:
            rand_img = self.transform(rand_img)
        return [data_img, category, text_feature, rand_img]

    def get_features(self):
        return self.text_features
    