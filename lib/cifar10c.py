import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image

from torchvision import datasets






class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None):
        
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )

        corruptions = [line.rstrip('\n') for line in open("/home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/domainbed/lib/corruptions.txt")]
        assert name in corruptions

        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)