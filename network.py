import torch.nn as nn
import torch.nn.functional as F
import torch

# Follow the hacks in https://github.com/soumith/ganhacks
class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()

        self.conv_text = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )

        self.conv_noise = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )

        self.network_concated = nn.Sequential(
        #   nn.ConvTranspose2d(nz*2, ngf*4, 4, 1, 0, bias=False),
        #   nn.BatchNorm2d(ngf*4),
        #   nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            
            # HACKS
            # nn.Dropout(p=0.5),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
  
    def forward(self, text, noise):
        x = self.conv_text(text)
        y = self.conv_noise(noise)
        input_concated = torch.cat([x,y],1)
        output = self.network_concated(input_concated)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc, nz, ndf):
        super(Discriminator, self).__init__()
        self.conv_image = nn.Sequential(
                nn.Conv2d(nc, int(ndf/2), 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_text = nn.Sequential(
                nn.Conv2d(nz, int(ndf/2), 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
        )

        self.network_concated = nn.Sequential(
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                # Weakening the discriminator with dropout
                nn.Dropout(p=0.5),

                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, image, text):
        x = self.conv_image(image)
        y = self.conv_text(text)
        input_concated = torch.cat([x,y],1)
        output = self.network_concated(input_concated)
        return output.view(-1, 1).squeeze(1)
