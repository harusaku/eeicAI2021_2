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
from utils import get_data_loader, generate_images, save_gif, txt2list, weight_init


# import clip
### To-do: replace all "noise"
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clipdraw dcgan')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--ndf', type=int, default=32, help='Number of features to be used in Discriminator network')
    parser.add_argument('--ngf', type=int, default=32, help='Number of features to be used in Generator network')
    parser.add_argument('--nv', type=int, default=512, help='Size of a CLIP Embedded Tensor') # Changed to the size of the embedded tensor from CLIP.
    # Changed from --nz to --nv
    parser.add_argument('--d-lr', type=float, default=0.00001, help='Learning rate for the discriminator')
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
    # label_ls = ['airplane', 'apple', 'banana']# ['airplane', 'apple', 'banana']
    label_ls = txt2list('categories')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)
    
    # Gather MNIST Dataset    
    train_loader, test_features = get_data_loader(opt.batch_size, label_ls, device)

    # Define Discriminator and Generator architectures
    netG = Generator(opt.nc, opt.nv, opt.ngf).to(device)
    netD = Discriminator(opt.nc, opt.nv, opt.ndf).to(device)

    # loss function
    criterion = nn.BCELoss()

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr)
    optimizerD_slow = optim.Adam(netD.parameters(), lr=opt.d_lr*0.25) # Fine tuning the lr of D(fake)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr)
    

    # initialize other variables
    real_label = 1
    fake_label = 0
    num_batches = len(train_loader)
    # fixed_noise = torch.randn(opt.num_test_samples, 100, 1, 1, device=device)
    
    ### CLIP Features
    # test_ls = ['airplane', 'apple', 'banana', 'bicycle', 'airplane', 'apple', 'banana', 'bicycle', 'airplane', 'apple', 'banana', 'bicycle', 'airplane', 'apple', 'banana', 'bicycle']
    # clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # tokenized_text = clip.tokenize(test_ls).to(device)
    # with torch.no_grad():
    #     test_features = clip_model.encode_text(tokenized_text)
    
    if test_features == None:
        print('No feature')
        os._exit(1)
    test_features = test_features.reshape((len(label_ls), 512, 1, 1)).float().to(device)
    test_noise = torch.zeros(len(label_ls), opt.nv, 1, 1, device=device)

    for epoch in range(opt.num_epochs):
        for i, (real_images, _, text_features, random_images) in enumerate(train_loader):
            # Normalize real_images to (-1,1)
            real_images = real_images*2-1
            real_images = real_images.to(device)
            random_images = random_images*2-1
            random_images = random_images.to(device)
            bs = real_images.shape[0]
            input_features = text_features.reshape((bs, opt.nv, 1, 1))
            broadcasted_features = torch.zeros(bs, opt.nv, 28, 28, device = device)
            discriminator_features = broadcasted_features + input_features # To allow conv inside netD
            
            ##############################
            #   Training discriminator   #
            ##############################
            netD.zero_grad()

            # REAL
            label = torch.full((bs,), real_label, device=device)
            output1 = netD(real_images, discriminator_features)
            lossD_real = criterion(output1, label.type(torch.float))
            lossD_real.backward()
            lossD_real = torch.mean(output1)
            D_x = output1.mean().item()
            
            # FAKE
            # TRICK2 Decaying noise:
            # noise = torch.randn(bs, opt.nv, 1, 1, device=device)*(1-epoch/opt.num_epochs)
            noise = torch.randn(bs, opt.nv, 1, 1, device=device)
            fake_images = netG(input_features, noise)
            label.fill_(fake_label)
            output2 = netD(fake_images.detach(), discriminator_features)
            lossD_fake = criterion(output2, label.type(torch.float))
            lossD_fake.backward()
            D_G_z1 = output1.mean().item() 
            optimizerD_slow.step()
            
            # Train again with real & real-but-wrong images
            netD.zero_grad()
            
            label = torch.full((bs,), real_label, device=device)
            output1 = netD(real_images, discriminator_features)
            lossD_real = criterion(output1, label.type(torch.float))
            lossD_real.backward()
            D_x = output1.mean().item()
            optimizerD.step()
            
            label.fill_(fake_label)
            output2 = netD(random_images, discriminator_features)
            lossD_wrong = criterion(output2, label.type(torch.float))
            lossD_wrong.backward()
            D_G_z1 = output1.mean().item() 
            optimizerD.step()

            ##########################
            #   Training generator   #
            ##########################
            for _ in range(4):
                netG.zero_grad()
                label.fill_(real_label)
                noise = torch.randn(bs, opt.nv, 1, 1, device=device)
                fake_images = netG(input_features, noise)
                output = netD(fake_images, discriminator_features)
                lossG = criterion(output, label.type(torch.float))
                lossG.backward()
                D_G_z2 = output.mean().item()
            optimizerG.step()

            if (i+1)%50 == 0:
                print('Epoch [{}/{}], step [{}/{}], d_loss_real: {:.4f}, d_loss_fake: {:.4f},  d_loss_wrong: {:.4f}, g_loss: {:.4f}'.format(epoch+1, opt.num_epochs, 
                      i+1, num_batches, lossD_real.item(), lossD_fake.item(), lossD_wrong.item(), lossG.item()))
        
        print('Epoch[{}/{}]'.format(epoch+1, opt.num_epochs))
        netG.eval()
        generate_images(epoch, opt.output_path, label_ls, test_features, test_noise, opt.num_test_samples, netG, device, use_fixed=opt.use_fixed)
        # Save model
        os.makedirs(opt.output_path + "model/", exist_ok=True)
        path = opt.output_path + "model/model_" + str(epoch + 1)
        torch.save(netG.state_dict(), path)

        netG.train()

    # Save gif:
    save_gif(opt.output_path, opt.fps)