import argparse
import logging
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

from network import Generator, Discriminator
# from utils import get_data_loader, generate_images, save_gif, txt2list
from utils_tmp import get_data_loader, generate_images, save_gif, txt2list


import clip
### To-do: replace all "noise"
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGANS MNIST')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--ndf', type=int, default=32, help='Number of features to be used in Discriminator network')
    parser.add_argument('--ngf', type=int, default=32, help='Number of features to be used in Generator network')
    parser.add_argument('--nv', type=int, default=512, help='Size of a CLIP Embedded Tensor') # Changed to the size of the embedded tensor from CLIP.
    # Changed from --nz to --nv
    parser.add_argument('--d-lr', type=float, default=0.0002, help='Learning rate for the discriminator')
    parser.add_argument('--g-lr', type=float, default=0.0002, help='Learning rate for the generator')
    parser.add_argument('--nc', type=int, default=1, help='Number of input channels. Ex: for grayscale images: 1 and RGB images: 3 ')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-test-samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--output-path', type=str, default='./results/', help='Path to save the images')
    parser.add_argument('--fps', type=int, default=5, help='frames-per-second value for the gif')
    parser.add_argument('--use-fixed', action='store_true', help='Boolean to use fixed noise or not')

    opt = parser.parse_args()
    print(opt)
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path, exist_ok=True)

    # Multi labels
    label_ls = ['airplane', 'apple']# ['airplane', 'apple', 'banana']

    # Gather MNIST Dataset    
    train_loader = get_data_loader(opt.batch_size, label_ls)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    # Define Discriminator and Generator architectures
    netG = Generator(opt.nc, opt.nv, opt.ngf).to(device)
    netD = Discriminator(opt.nc, opt.ndf).to(device)

    # loss function
    criterion = nn.BCELoss()

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr)
    
    # initialize other variables
    real_label = 1
    fake_label = 0
    num_batches = len(train_loader)
    # fixed_noise = torch.randn(opt.num_test_samples, 100, 1, 1, device=device)
    
    ### CLIP Features
    model, preprocess = clip.load("ViT-B/32", device=device)
    tokenized_text = clip.tokenize(label_ls).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokenized_text)
    dict_text_features = {}
    for i in range(len(label_ls)):
        dict_text_features[label_ls[i]] = text_features[i].cpu().numpy() # convert to tensors to numpy arrays

    for epoch in range(opt.num_epochs):
        for i, (real_images, labels) in enumerate(train_loader):
            bs = real_images.shape[0]
            ##############################
            #   Training discriminator   #
            ##############################

            netD.zero_grad()
            real_images = real_images.to(device)
            label = torch.full((bs,), real_label, device=device)

            output = netD(real_images)
            lossD_real = criterion(output, label.type(torch.float))
            lossD_real.backward()
            D_x = output.mean().item()

            tensor_text_features = torch.tensor([ dict_text_features[la] for la in labels ], device = device).reshape((bs, 512, 1, 1)).float()
            fake_images = netG(tensor_text_features)
            # noise = torch.randn(bs, opt.nv, 1, 1, device=device)
            # fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach())
            lossD_fake = criterion(output, label.type(torch.float))
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            ##########################
            #   Training generator   #
            ##########################

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_images)
            lossG = criterion(output, label.type(torch.float))
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if (i+1)%10 == 0:
                print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'.format(epoch+1, opt.num_epochs, 
                                                            i+1, num_batches, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
        netG.eval()
        generate_images(epoch, opt.output_path, label_ls, tensor_text_features, opt.num_test_samples, netG, device, use_fixed=opt.use_fixed)
        netG.train()

    # Save gif:
    save_gif(opt.output_path, opt.fps)