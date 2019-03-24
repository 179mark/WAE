#%matplotlib inline
import matplotlib.pyplot as plt
import pickle
import numpy as np
#from tqdm import tqdm
#from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.nn.modules.loss import MSELoss
from torchvision.utils import save_image



class DataSampler(BatchSampler):
    def __init__(self, batch_size, num_classes, labels):
        self.num_classes = num_classes
        self.N = batch_size
        self.labels = labels

    def __iter__(self):
        num_yielded = 0
        while num_yielded < self.labels.size()[0]:
            batch = torch.randint(high=self.labels.size()[0], size=(self.N,)).long()
            num_yielded += self.N
            yield batch


input_size = 784
num_classes = 10
batch_size = 256

train_dataset = dsets.MNIST(root='./MNIST/', 
                                   train=True, 
                                   transform=transforms.ToTensor(),
                                   download=True)

test_dataset = dsets.MNIST(root='./MNIST/', 
                                  train=False, 
                                  transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_sampler=DataSampler(batch_size=batch_size, num_classes=num_classes, labels=train_dataset.train_labels), 
                                          shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                         batch_sampler=DataSampler(batch_size=batch_size, num_classes=num_classes, labels=test_dataset.test_labels), 
                                         shuffle=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(1024, 8),
        )

    def forward(self, x):
        output = self.cnn1(x)
        return output

class DeFlatten(nn.Module):
    def forward(self, x):
        return x.view( -1, 1024, 7, 7 )

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Linear(8, 7 * 7 * 1024),
            nn.ReLU(),
            DeFlatten(),
            nn.ConvTranspose2d(1024, 512, 4),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 1, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.cnn1(x)
        return output


def imq_kernel(X: torch.Tensor, Y: torch.Tensor, h_dim: int):
    batch_size = X.size(0)

    p2_norm_x = X.pow(2).sum(1).unsqueeze(0)
    norms_x = X.sum(1).unsqueeze(0)
    prods_x = torch.mm(norms_x, norms_x.t())
    dists_x = p2_norm_x + p2_norm_x.t() - 2 * prods_x

    p2_norm_y = Y.pow(2).sum(1).unsqueeze(0)
    norms_y = X.sum(1).unsqueeze(0)
    prods_y = torch.mm(norms_y, norms_y.t())
    dists_y = p2_norm_y + p2_norm_y.t() - 2 * prods_y

    dot_prd = torch.mm(norms_x, norms_y.t())
    dists_c = p2_norm_x + p2_norm_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        #if torch.cuda.is_available():
        #    res1 = (1 - torch.eye(batch_size).cuda()) * res1
        #else:
        res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats





encoder, decoder = Encoder(), Decoder()
encoder.train()
decoder.train()

opt_e = torch.optim.Adam( encoder.parameters(), lr=0.0005 )
opt_d = torch.optim.Adam( decoder.parameters(), lr=0.0005 )

mse = MSELoss()


for epoch in range(50):
    for x, target in train_loader:
        opt_e.zero_grad()
        opt_d.zero_grad()

        z = encoder(x)
        x_r = decoder(z)

        z_f = torch.randn(x.size()[0], 8)

        mmd_loss = imq_kernel(z, z_f, h_dim=8)
        mmd_loss = mmd_loss.mean()

        loss = mse(x, x_r) - mmd_loss
        loss.backward()

        opt_e.step()
        opt_d.step()
        
        print(epoch, loss.data)
        
    x = next(iter(test_loader))[0]
    z = encoder(x)
    x_r = decoder(torch.randn_like(z))

    save_image(x.view(-1, 1, 28, 28), './x_%d.png' % (epoch) )
    save_image(x_r.data.view(-1, 1, 28, 28), './x_r_%d.png' % (epoch) )

    with open('encoder_%d.pickle' % (epoch) , 'wb') as f:
        pickle.dump(encoder, f)
    with open('decoder_%d.pickle' % (epoch) , 'wb') as f:
        pickle.dump(decoder, f)
