from genericpath import exists
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import itertools
import imageio
import natsort
from glob import glob
import os

# from dataset import MyDataset
from dataset_tmp import MyDataset

#######################################
## 使うカテゴリの名前を代入          ##
#######################################
# category = "apple"
# category_ls = ['airplane', 'apple', 'banana']
features = None
def get_data_loader(batch_size, category_ls, device='cpu'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))])
    train_dataset = MyDataset(category_ls,"./filtered_data/", transform=transform, device = device)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, train_dataset.get_features()


### Added embedded features as a parameter
def generate_images(epoch, path, category_ls, tensor_text_features, num_test_samples, netG, device, use_fixed=False):
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
    path += ('text_features' + "/")

    generated_fake_images = netG(tensor_text_features)

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
    
    os.makedirs(path, exist_ok=True)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')

def save_gif(path, fps):
    path += ('text_features' + "/")
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    os.makedirs("gif/", exist_ok=True)
    imageio.mimsave("gif/"+ 'features' +'_animated.gif', gif, fps=fps)

def txt2list(filename):
    f = open(filename + ".txt", "r", encoding="UTF-8")
    ret = []
    for line in f:
        ret.append(line.rstrip())
    return ret
    