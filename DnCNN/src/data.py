import os
import torch
import torch.utils.data as td
import torchvision as tv
import pandas as pd
import numpy as np
from PIL import Image




class NoisyBSDSDataset(td.Dataset):

    def __init__(self, root_dir, mode='train', image_size=(180, 180), sigma=30):
        super(NoisyBSDSDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyBSDSDataset(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        clean = Image.open(img_path).convert('RGB')
        # random crop
        i = np.random.randint(clean.size[0] - self.image_size[0])
        j = np.random.randint(clean.size[1] - self.image_size[1])

        clean = clean.crop([i, j, i+self.image_size[0], j+self.image_size[1]])
        transform = tv.transforms.Compose([
            # convert it to a tensor
            tv.transforms.ToTensor(),
            # normalize it to the range [−1, 1]
            tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        clean = transform(clean)

        noisy = clean + 2 / 255 * self.sigma * torch.randn(clean.shape)
        return noisy, clean
    
class NoisyBSDSDatasetMult(td.Dataset):

    def __init__(self, root_dir, mode='train', image_size=(180, 180), sigma=30):
        super(NoisyBSDSDatasetMult, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyBSDSDatasetMult(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        clean = Image.open(img_path).convert('RGB')
        # random crop
        i = np.random.randint(clean.size[0] - self.image_size[0])
        j = np.random.randint(clean.size[1] - self.image_size[1])

        clean = clean.crop([i, j, i+self.image_size[0], j+self.image_size[1]])
        transform = tv.transforms.Compose([
            # convert it to a tensor
            tv.transforms.ToTensor(),
            # normalize it to the range [−1, 1]
            tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        clean = transform(clean)


        L = (255.0 / self.sigma) ** 2

        noise = torch.distributions.Gamma(
            concentration=torch.tensor(L, device=clean.device),
            rate=torch.tensor(L, device=clean.device)
        ).sample(clean.shape)

        noisy = clean * noise

        return noisy, clean
    

class NoisyBSDSDatasetMultCorrected(td.Dataset):
    """
    # Noisy Dataset mit Gamma-verteiltem multiplikativem Rauschen,
    # parametrisiert über M (Anzahl Looks) wie im MIDAL-Algorithmus
    """

    def __init__(self, root_dir, mode='train', image_size=(180, 180), sigma=30):
        super(NoisyBSDSDatasetMultCorrected, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        clean = Image.open(img_path).convert('RGB')
        
        # Random crop
        i = np.random.randint(clean.size[0] - self.image_size[0])
        j = np.random.randint(clean.size[1] - self.image_size[1])
        clean = clean.crop([i, j, i+self.image_size[0], j+self.image_size[1]])
        
        # Zu Tensor [0, 1]
        clean_01 = tv.transforms.ToTensor()(clean)
        
        # Rauschen in [0, 1] anwenden 
        L = (255.0 / self.sigma) ** 2
        noise = torch.distributions.Gamma(
            concentration=torch.tensor(L),
            rate=torch.tensor(L)
        ).sample(clean_01.shape)
        
        noisy_01 = clean_01 * noise
        noisy_01 = torch.clamp(noisy_01, 0, 1)
        
        # DANN beide auf [-1, 1] normalisieren
        clean = clean_01 * 2 - 1
        noisy = noisy_01 * 2 - 1
        
        return noisy, clean
    

    
