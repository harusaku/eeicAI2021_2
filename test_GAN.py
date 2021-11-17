import torch
import torch.nn as nn
import imageio
import os
import matplotlib.pyplot as plt
import clip

from network import Generator

device = "cpu"
model_path = "./model/model_20categories"
G_feature = 32
netG = Generator(1, 512, G_feature).to(device)
netG.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
test_noise = torch.zeros(1, 512, 1, 1, device=device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

while True:
    input_text = input("What to draw?\n")
    tokenized_text = clip.tokenize([input_text]).to(device)
    with torch.no_grad():
        feature = clip_model.encode_text(tokenized_text)
    feature = feature.reshape((1, 512, 1, 1)).float().to(device)
    generated_images = netG(feature, test_noise)
    image = (generated_images[0]+1) / 2
    plt.imshow(image.data.cpu().numpy().reshape(28,28), cmap='Greys')
    os.makedirs("test/", exist_ok=True)
    plt.savefig("test/"+input_text+".png")
    print("Result generated at " + "./test/"+input_text+".png\n\n")




### Added embedded features as a parameter
def generate_images(epoch, path, category_ls, tensor_text_features, noise, num_test_samples, netG, device, use_fixed=False):
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
    path += ('text_features' + "/")

    generated_fake_images = netG(tensor_text_features, noise)

    # restore to (-1,1) to (0,1)
    generated_fake_images = (generated_fake_images+1)/2
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6,6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k//4
        j = k%4
        ax[i,j].cla()
        ax[i,j].set_title(category_ls[k])
        ax[i,j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28,28), cmap='Greys')
    label = 'Epoch_{}'.format(epoch+1)
    
    os.makedirs(path, exist_ok=True)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')