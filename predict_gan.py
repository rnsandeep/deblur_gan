import argparse
import os, cv2
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from tqdm import tqdm

n_epochs = 1
data_dir = 'distract'
batch_size=1
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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.layer2 = nn.LeakyReLU()
        self.layer3 = nn.Conv2d(32, 3, 3, 1, 1)
        self.layer4 = nn.LeakyReLU() #Sigmoid() #LeakyReLU()
        layers = [self.layer1, self.layer2, self.layer3, self.layer4] #, self.layer5, self.layer6, self.layer7, self.layer8]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


def load_model(path):
    model = torch.load(path)
    return model


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.299, 0.225]


def datatransforms(crop_size):
    print("mean and standard deviation:",mean,std)
    data_transforms = {
      'train': transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor(),
#        transforms.Normalize( mean, std)
      ]),
      'val': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
#        transforms.Normalize(mean, std)
      ]),
      'test': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
#        transforms.Normalize(mean, std)
      ]),

    }
    return data_transforms

def degrade_image(image):
    image = tensor_to_cv2(image)
    SHIFT = 2
    to_swap = np.random.choice([False, True], image.shape[:2], p=[.8, .2])
    swap_indices = np.where(to_swap[:-SHIFT] & ~to_swap[SHIFT:])
    swap_vals = image[swap_indices[0] + SHIFT, swap_indices[1]]
    image[swap_indices[0] + SHIFT, swap_indices[1]] = image[swap_indices]
    image[swap_indices] = swap_vals
    return np.transpose(image, (2, 0, 1))

def tensor_to_cv2(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


data_transforms = datatransforms(299)

print("testing images from folder:", datasets.ImageFolder(os.path.join(data_dir, 'test')))

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}

dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             num_workers=16)
              for x in ['test']}


generator = Generator()
checkpoint = load_model(sys.argv[1]) #'generator_checkpoint.pth.tar')
generator.load_state_dict(checkpoint['state_dict'])
generator = generator.cuda()


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
output_dir = 'results'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def test():
    phase = 'test'
    losses = []
    for i, (imgs, y) in enumerate(tqdm(dataloader[phase])):
       
        # degrading images using given function.
        deg_imgs = []
        for img in imgs:
            deg_img = degrade_image(img.clone())
            deg_imgs.append(torch.tensor(deg_img))
        deg_imgs = torch.stack(deg_imgs)

        x = Variable(deg_imgs.type(Tensor)) # degraded images are input.
        x_upg = generator(x) # expected output will be upgraded images. # prediction.

        # write generated images concatenated with degraded image and original images to results folder.

        criterion = nn.MSELoss()
        loss = criterion(x_upg.cpu(), imgs).item()

#        print("loss for all images:", loss)
        losses.append(loss)


        for idx, pred in enumerate(x_upg):
            pred = tensor_to_cv2(pred)
            deg = tensor_to_cv2(deg_imgs[idx])
            org = tensor_to_cv2(imgs[idx])

            pred = (pred * 255).astype(np.uint8)
            deg = (deg * 255).astype(np.uint8)
            org = (org * 255).astype(np.uint8)

            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            deg = cv2.cvtColor(deg, cv2.COLOR_RGB2BGR)
            org = cv2.cvtColor(org, cv2.COLOR_RGB2BGR)

            cv2_deg = cv2.putText(deg, "Degraded", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2_pred = cv2.putText(pred, "Prediction", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2_org = cv2.putText(org, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            concat = np.concatenate((cv2_pred, cv2_deg), axis=1)
            concat = np.concatenate((concat, cv2_org), axis=1)

            cv2.imwrite(os.path.join(output_dir, str(i*batch_size+idx)+'.jpg'), concat)
    losses = np.array(losses)
    print("average loss mse:", np.mean(losses))
     

test()
print("concatenated results are written in results folder")
