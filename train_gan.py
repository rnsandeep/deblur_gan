import argparse
import os, cv2
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets, models
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

n_epochs = 100
data_dir = sys.argv[1]
batch_size=4
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
img_size=250
channels = 3
interval = 100

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.299, 0.225]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.layer2 = nn.LeakyReLU()
        self.layer3 = nn.Conv2d(32, 3, 3, 1, 1)
        self.layer4 = nn.LeakyReLU() #LeakyReLU()
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out        

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.net = nn.Conv2d(3, 32, 1, 1)

    def forward(self, x):
        out = self.net(x)
        return out #self.sigmoid(out)


class Discriminator():
    def model(self):
       num_classes = 2
       model_ft = models.inception_v3(pretrained=True)
       num_ftrs = model_ft.AuxLogits.fc.in_features
       model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
       num_ftrs = model_ft.fc.in_features
       model_ft.fc = nn.Linear(num_ftrs,num_classes)
       discriminator = model_ft.cuda()
       return discriminator
   
   
# Initialize generator and discriminator
generator = Generator()
discriminator =  Discriminator().model()  #Discriminator().model()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
#discriminator.apply(weights_init_normal)
un = UnNormalize(mean, std)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.299, 0.225]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def datatransforms(crop_size):
    print("mean and standard deviation:",mean,std)
    data_transforms = {
      'train': transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor(),
       # transforms.Normalize( mean, std)
      ]),
      'val': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
       # transforms.Normalize(mean, std)
      ]),
      'test': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
       # transforms.Normalize(mean, std)
      ]),

    }
    return data_transforms

data_transforms = datatransforms(299)


print(datasets.ImageFolder(os.path.join(data_dir, 'train')))

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']}

dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             num_workers=16)
              for x in ['train']}

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def tensor_to_cv2(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def degrade_image(image):
    image = tensor_to_cv2(image)
    SHIFT = 2
    to_swap = np.random.choice([False, True], image.shape[:2], p=[.8, .2])
    swap_indices = np.where(to_swap[:-SHIFT] & ~to_swap[SHIFT:])
    swap_vals = image[swap_indices[0] + SHIFT, swap_indices[1]]
    image[swap_indices[0] + SHIFT, swap_indices[1]] = image[swap_indices]
    image[swap_indices] = swap_vals
    return np.transpose(image, (2, 0, 1))

#  Training

for epoch in range(n_epochs):
    phase = 'train'
    for i, (imgs, label) in enumerate(dataloader[phase]):
        deg_imgs = []
        for img in imgs:
            deg_img = degrade_image(img.clone()) #np.transpose(degrade_image(np.transpose(img, (1, 2, 0))), (2, 0, 1))
            deg_imgs.append(torch.tensor(deg_img))

        deg_imgs = torch.stack(deg_imgs)

        y = Variable(imgs.type(Tensor))  # original images are target.
        x = Variable(deg_imgs.type(Tensor)) # degraded images are input.

        # Loss  Discrminator

        optimizer_D.zero_grad()

        x_upg = generator(x)
        _, d_upg = discriminator(x_upg) # generator trying to be real. # should be minimized.
        _ , d_org = discriminator(y) # should be maximized.
        target = torch.ones([batch_size], dtype=torch.int64).cuda() # target should be one(real image) for both generator and discrimantor.

        loss = nn.MSELoss() #reduce=False)
        loss_pixelwise = nn.L1Loss()
        loss_d = -1*loss(d_upg, d_org).sum() / batch_size + -1*loss_pixelwise(d_upg, d_org).sum()/ batch_size

        loss_d.cuda()
        print("loss of discriminator:", loss_d.item())

        loss_d.backward() #retain_graph=True)
        optimizer_D.step()

        #  Loss Generator

        optimizer_G.zero_grad()

        x_upg = generator(x)
        _, d_upg = discriminator(x_upg) # should be maximized. Generator wants to make these images real.
        _ , d_org = discriminator(y)
        target = torch.ones([4], dtype=torch.int64).cuda()

        loss_g = loss(d_upg, d_org).sum() / batch_size + loss_pixelwise(d_upg, d_org).sum()/ batch_size#-1*torch.mean(d_upg.double())

        print("generator loss:", loss_g.item())
        loss_g.cuda()

        loss_g.backward()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] ---------------------------------[D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), loss_d.item(), loss_g.item())
        )
        if i%1 ==0 :
           save_checkpoint({'state_dict': generator.state_dict()}, False, str(i)+'generator_checkpoint.pth.tar')

