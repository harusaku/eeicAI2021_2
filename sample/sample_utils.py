from genericpath import exists
import torch
import matplotlib.pyplot as plt
from torch.serialization import load
from torchvision import datasets, transforms
import math
import itertools
import imageio
import natsort
from glob import glob
import os

from sample_dataset import MyDataset

#######################################
## 使うカテゴリの名前を代入          ##
#######################################

def get_data_loader(batch_size, category, load_path):
    load_path = load_path.replace("/","").replace(".","")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))])
    train_dataset = MyDataset(category,load_path, transform=transform)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def generate_images(load_path, category, epoch, path, fixed_noise, num_test_samples, netG, device, use_fixed=False):
    load_path = load_path.replace("/","").replace(".","")
    z = torch.randn(num_test_samples, 100, 1, 1, device=device)
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
    path += (load_path + "/" + category + "/")
  
    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
        title = 'Fixed Noise'
        os.makedirs(path, exist_ok=True)
        
    else:
        generated_fake_images = netG(z)
        path += 'variable_noise/'
        title = 'Variable Noise'
        os.makedirs(path, exist_ok=True)
  
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6,6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k//4
        j = k%4
        ax[i,j].cla()
        ax[i,j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28,28), cmap='Greys')
    label = 'Epoch_{}'.format(epoch+1)

    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')

def save_gif(load_path, category, path, fps, fixed_noise=False):
    load_path = load_path.replace("/","").replace(".","")
    path += load_path + "/" + category +"/"
    if fixed_noise==True:
        path += 'fixed_noise/'
        os.makedirs(path, exist_ok=True)
    else:
        path += 'variable_noise/'
        os.makedirs(path, exist_ok=True)
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    os.makedirs("gif/sample/"+load_path+"/", exist_ok=True)
    imageio.mimsave("gif/sample/"+load_path+"/"+category +'_animated.gif', gif, fps=fps)

def txt2list(filename):
    f = open(filename + ".txt", "r", encoding="UTF-8")
    ret = []
    for line in f:
        ret.append(line.rstrip())
    return ret
    