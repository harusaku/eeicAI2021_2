import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from text2list import text2list

class MyDataset(Dataset):
	def __init__(self, root=None, train=True, transform=None):
		self.root = root
		self.transform = transform
		mode = 'train' if train else 'test'
		# categoryの配列を取得
		self.categories = text2list("categories.txt")
		# 画像パスとそのcategoryのセットをself.all_dataに入れる
		self.all_data = []

		for category in self.categories:
			data = np.load(self.root+ category.replace(" ", "_") + ".npy")
			category_data = [[data_i.reshape(28,28), category] for data_i in data]

			self.all_data.extend(category_data)

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