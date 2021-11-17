from genericpath import exists
import torch
import matplotlib.pyplot as plt
import imageio
import natsort
from glob import glob
import os
import sys


def save_gif(category, path, fps, fixed_noise=False):
    path += "filtered_data/" + category +"/"
    if fixed_noise==True:
        path += 'fixed_noise/'
        os.makedirs(path, exist_ok=True)
    else:
        path += 'variable_noise/'
        os.makedirs(path, exist_ok=True)
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for i in range(len(images)):
        if((i+1)%10==0):
            gif.append(imageio.imread(images[i]))
    os.makedirs("gif/sample/filtered_data", exist_ok=True)
    imageio.mimsave("gif/sample/filtered_data/"+category +'_animated_short.gif', gif, fps=fps)

if __name__ == '__main__':
    args = sys.argv
    save_gif(args[1], "./results/", 5)