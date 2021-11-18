import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
class MyDataset(Dataset):
	def __init__(self, category, root=None, train=True, transform=None):
		self.root = root + "/"
		self.transform = transform
		mode = 'train' if train else 'test'
		# TODO class辞書の定義
		self.category = category
		# TODO 画像パスとそのラベルのセットをself.all_dataに入れる
		self.all_data = []

		data = np.load(self.root+ self.category + ".npy")
		self.data = [[data_i.reshape(28,28),self.category] for data_i in data]

	def __len__(self):
		#データセットの数を返す関数
		return len(self.data)
		
	def __getitem__(self, idx):
		# TODO 画像とラベルの読み取り
		#self.all_dataを用いる
		data_img = Image.fromarray(self.data[idx][0]).convert("L")
		if self.transform is not None:
			data_img = self.transform(data_img)
		label = self.data[idx][1]
		return [data_img, label]