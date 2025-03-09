# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import math
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
# from torchvision.datasets import MNIST, ImageFolder,  USPS, SVHN, MNISTM, SYN
# from torchvision.datasets import PACSDataset as pacs
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10 as cifar10
from torchvision.transforms.functional import rotate
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torchvision.transforms.functional import InterpolationMode
#from domainbed.lib.cifar10c import CIFAR10C as cifar10c

# MNISTM, SYN,
# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    "DIGITS",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    "ImageNet_9",
    "ImageNet_C",
    "ImageNet_val",
    "ImageNet_V2",
    "ImageNet_R",
    "ImageNet_S",
    "CIFAR10",
    "CIFAR10C"
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    "Cue_conflicts",
    "PACS_Custom"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

#me# return the number of available domains
def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

def name_environments(dataset_name):
    return get_dataset_class(dataset_name).ENVIRONMENTS

class GreyToColor(object):
    """Convert Grey Image label to binary
    """

    def __call__(self, image):
        if len(image.size()) == 3 and image.size(0) == 1:
            return image.repeat([3, 1, 1])
        elif len(image.size())== 2:
            return
        else:
            return image

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class IdentityTransform():
    """do nothing"""

    def __call__(self, image):
        return image    


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override
    

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]: #domains list
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            ) # append one dataset for the considered domain. say, 16 images, along with their labels

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))
        

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
   
    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)
    
class CIFAR10C(MultipleDomainDataset):
    ENVIRONMENTS = sorted([line.rstrip('\n') for line in open("/home/kavindya/data/Model/TFS-ViT_Token-level_Feature_Stylization/domainbed/lib/corruptions_ME_ADA.txt")])
    N_STEPS = 5000           
    CHECKPOINT_FREQ = 300
    INPUT_SHAPE = (3, 32, 32)
    def __init__(self, root, test_envs, hparams):
        root = os.path.join(root, "CIFAR-10-C/")
        super().__init__()
        self.datasets = []
        envs = self.ENVIRONMENTS

        # if hparams["backbone"]=="ViTBase":
        #     MEAN = [0.5, 0.5, 0.5]
        #     STD = [0.5, 0.5, 0.5]
        # else:
        #     MEAN = [0.485, 0.456, 0.406]
        #     STD = [0.229, 0.224, 0.225]
        
        # if (hparams["backbone"]=="ViTBase") or (hparams["backbone"]=="DeiTBase"):
        #     transform = transforms.Compose([
        #         transforms.Resize(size=math.floor(224/0.9),interpolation=InterpolationMode.BICUBIC),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=MEAN, std=STD)
        #     ])
        # else:

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize([0.5] * 3, [0.5] * 3)
                ])

        for corruption in envs:
            datast = cifar10c(root, corruption, transform=test_transform)
            self.datasets.append(datast)

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 10
        
        
class CIFAR10(MultipleDomainDataset):
    ENVIRONMENTS = ["real_train","real_val"]
    N_STEPS = 39882          
    CHECKPOINT_FREQ = 391
    INPUT_SHAPE = (3, 32, 32)
    def __init__(self, root, test_envs, hparams):
        root = os.path.join(root, "cifar/")
        super().__init__()
        self.datasets = []
        envs = self.ENVIRONMENTS  

        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor() ,
                #transforms.ConvertImageDtype(torch.float),
                transforms.Normalize([0.5] * 3, [0.5] * 3) if hparams["normalization"] else transforms.Lambda(lambda x: x)
            ])

        hparams["mean_std"]=[[0.5] * 3, [0.5] * 3]        

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)])

        
        self.datasets.append(cifar10(root="/media/SSD2/Dataset/cifar",train=True, transform=train_transform, download=False))
        print("Training dataset completed")

        self.datasets.append(cifar10(root="/media/SSD2/Dataset/cifar",train=False, transform=test_transform, download=False))
        print("Validation dataset completed")

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 10

class DIGITS(MultipleDomainDataset):
    ENVIRONMENTS = ['MNIST','MNIST_VAL','MNIST-M', 'SVHN','USPS','SYN'] #
    INPUT_SHAPE = (3, 32, 32)
    N_STEPS = 10000           
    CHECKPOINT_FREQ = 250
    STEPS_PER_EPOCH = 250
    def __init__(self, root, test_envs, hparams):
        root = os.path.join(root, "digits/")
        super().__init__()
        self.datasets = []

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((32,32), scale=(0.5, 1)),
           
        ])

        # train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop((32,32), scale=(0.5, 1)),
        #     transforms.ToTensor(),
        #      GreyToColor(),
        #     transforms.Normalize(
        #         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])

        transform  = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.CenterCrop((32,32)),
        transforms.ToTensor(),
        GreyToColor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        #loading MNIST DATASET
        original_dataset_tr = MNIST(root, train=True, transform=train_transform, download=True)
        original_dataset_te = MNIST(root, train=False, transform=transform, download=True)

        original_dataset_tr.data = original_dataset_tr.data[:10000]
        original_dataset_tr.targets = original_dataset_tr.targets[:10000]

        # original_dataset_tr.data = torch.cat((original_dataset_te.data,original_dataset_tr.data))
        # original_dataset_tr.targets = torch.cat((original_dataset_te.targets,original_dataset_tr.targets))
        
        self.datasets.append(original_dataset_tr)
        self.datasets.append(original_dataset_te)

        #loading MNIST-M DATASET
        original_dataset_te = MNISTM(root, train=False, transform=transform, download=True)

        self.datasets.append(original_dataset_te)

        #loading SVHN
        original_dataset_te = SVHN(root, split="test", transform=transform, download=True)

        self.datasets.append(original_dataset_te)

        #loading USPS
        original_dataset_te = USPS(root, train=False, transform=transform, download=True)

        self.datasets.append(original_dataset_te)

        #loading SYN
        original_dataset_te = SYN(root, train=False, transform=transform, download=True)

        self.datasets.append(original_dataset_te)

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 10


class PACS_Custom(MultipleDomainDataset):
    CHECKPOINT_FREQ = 300
    N_STEPS = 2500
    ENVIRONMENTS = ['AAP', 'AR', 'C', 'S']
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        root = os.path.join(root, "PACS/Unsplit_data")
        environments = self.ENVIRONMENTS
        environments = sorted(environments) # list of all domains in the dataset, in sorted order
        
        if hparams["backbone"]=="ViTBase":
            MEAN = [0.5, 0.5, 0.5]
            STD = [0.5, 0.5, 0.5]
        else:
            MEAN = [0.485, 0.456, 0.406]
            STD = [0.229, 0.224, 0.225]

        hparams["mean_std"] = [MEAN,STD]

        # train transform
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224,
                interpolation=InterpolationMode.BILINEAR,antialias=True),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            #transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=MEAN, std=STD) if hparams["normalization"] else transforms.Lambda(lambda x: x)
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if hparams['data_augmentation'] and (i not in test_envs):
                print('[INFO] Doing Special Data Augmentation for Training')
                env_transform = augment_transform
            elif i not in test_envs:
                print('[INFO] NOT Doing Special Data Augmentation for Training')
                env_transform = transform
            else:
                
                hparams["normalization"]=True
                
                if hparams["normalization"]==True: 
                    print("++++++++ Using Normailztion for val/test data")
                else:
                    print("+++++++++++ Not Using Normailztion for val/test data")

                print('[INFO] Doing ImageNet validation Augmentation')
                if (hparams["backbone"]=="ViTBase") or (hparams["backbone"]=="DeiTBase"):
                    env_transform = transforms.Compose([
                        transforms.Resize(size=math.floor(224/0.9),
                                          interpolation=InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=MEAN, std=STD) if hparams["normalization"] else transforms.Lambda(lambda x: x)
                    ])
                else:
                
                    env_transform = transforms.Compose([
                    transforms.Resize(size=256,
                               interpolation=InterpolationMode.BILINEAR,antialias=True),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=MEAN, std=STD) if hparams["normalization"] else transforms.Lambda(lambda x: x)
                ])
                
                
                
            path = os.path.join(root, environment)
            env_dataset = pacs(root=path, transform=env_transform)
            self.datasets.append(env_dataset)
            
        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
        
class IdentityTransform:
    """A transform that returns the input data as is."""
    def __call__(self, x):
        return x

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()

        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments) # list of all domains in the dataset, in sorted order
        
        if hparams["backbone"]=="ViTBase":
            MEAN = [0.5, 0.5, 0.5]
            STD = [0.5, 0.5, 0.5]
        else:
            MEAN = [0.485, 0.456, 0.406]
            STD = [0.229, 0.224, 0.225]

        hparams["mean_std"] = [MEAN,STD]

        # train transform
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224,
                interpolation=InterpolationMode.BILINEAR,antialias=True),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            #transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=MEAN, std=STD) if hparams["normalization"] else IdentityTransform()
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                print('[INFO] Doing Special Data Augmentation for Training')
                env_transform = augment_transform
            elif i not in test_envs:
                print('[INFO] NOT Doing Special Data Augmentation for Training')
                env_transform = transform
            else:
                
                hparams["normalization"]=True
                
                if hparams["normalization"]==True: 
                    print("++++++++ Using Normailztion for val/test data")
                else:
                    print("+++++++++++ Not Using Normailztion for val/test data")

                print('[INFO] Doing ImageNet validation Augmentation')
                if (hparams["backbone"]=="ViTBase") or (hparams["backbone"]=="DeiTBase"):
                    env_transform = transforms.Compose([
                        transforms.Resize(size=math.floor(224/0.9),
                                          interpolation=InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=MEAN, std=STD) if hparams["normalization"] else IdentityTransform()
                    ])
                else:
                #     env_transform = transforms.Compose([
                #     transforms.Resize(size=256,
                #                 interpolation=InterpolationMode.BILINEAR,antialias=True),
                #     transforms.CenterCrop(224),
                #     transforms.ToTensor(),
                #     transforms.Normalize(
                #         mean=MEAN, std=STD)
                # ])
                    env_transform = transforms.Compose([
                    transforms.Resize(size=256,
                               interpolation=InterpolationMode.BILINEAR,antialias=True),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=MEAN, std=STD) if hparams["normalization"] else IdentityTransform()
                ])
                
                
                
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)
            self.datasets.append(env_dataset)
            
        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
        
class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

# class PACS(MultipleEnvironmentImageFolder):
#     # Photo domain should taken to first
#     CHECKPOINT_FREQ = 300
#     N_STEPS = 2500
#     ENVIRONMENTS = ['AAP_train', 'AAP_val', 'AR', 'C', 'S']
#     def __init__(self, root, test_envs, hparams):
#         self.dir = os.path.join(root, "PACS/Split_data")
#         super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    N_STEPS = 2500
    ENVIRONMENTS = ['AAP', 'AR', 'C', 'S']
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/Unsplit_data")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    N_STEPS = 8000
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "DomainNet/un")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class ImageNet_9(MultipleEnvironmentImageFolder):
    N_STEPS = 10000
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ['mixed_next',
                    'mixed_rand',
                    'mixed_same',
                    'no_fg',
                    'only_bg_b',
                    'only_bg_t',
                    'only_fg',
                    'original']
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "ImageNet_9/bg_challenge")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class Cue_conflicts(MultipleEnvironmentImageFolder):
    N_STEPS = 10000
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ['iid','texture']
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "Cue_conflicts_stimuli/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class ImageNet_C(MultipleEnvironmentImageFolder):
    N_STEPS = 10000
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = sorted([f.name for f in os.scandir("/media/SSD2/Dataset/Imagenet-C/corruption_severity") if f.is_dir() ])               
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "Imagenet-C/corruption_severity")
        
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class ImageNet(MultipleEnvironmentImageFolder):
    N_STEPS = 20019*20
    N_WORKERS = 16
    CHECKPOINT_FREQ = 5000
    ENVIRONMENTS = ["train","valid"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "ImageNet_val/JustCopy")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class ImageNet_V2(MultipleEnvironmentImageFolder):
    N_STEPS = 5000
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["eval"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "ImageNet_V2/data")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class ImageNet_S(MultipleEnvironmentImageFolder):
    N_STEPS = 5000
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["eval"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "ImageNet-S/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class ImageNet_R(MultipleEnvironmentImageFolder):
    N_STEPS = 5000
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["imagenet-r"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "ImageNet-R/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class ImageNet_200(MultipleEnvironmentImageFolder):
    N_STEPS = 5000
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["imagenet_val_for_imagenet_r"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join("/home/kavindya/data/Model/ImageNet_training/Dataset")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class ImageNet_val(MultipleEnvironmentImageFolder):
    N_STEPS = 5000
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["valid"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "ImageNet_val/validation")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
    
class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


# class WILDSEnvironment:
#     def __init__(
#             self,
#             wilds_dataset,
#             metadata_name,
#             metadata_value,
#             transform=None):
#         self.name = metadata_name + "_" + str(metadata_value)

#         metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
#         metadata_array = wilds_dataset.metadata_array
#         subset_indices = torch.where(
#             metadata_array[:, metadata_index] == metadata_value)[0]

#         self.dataset = wilds_dataset
#         self.indices = subset_indices
#         self.transform = transform

#     def __getitem__(self, i):
#         x = self.dataset.get_input(self.indices[i])
#         if type(x).__name__ != "Image":
#             x = Image.fromarray(x)

#         y = self.dataset.y_array[self.indices[i]]
#         if self.transform is not None:
#             x = self.transform(x)
#         return x, y

#     def __len__(self):
#         return len(self.indices)


# class WILDSDataset(MultipleDomainDataset):
#     INPUT_SHAPE = (3, 224, 224)
#     def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
#         super().__init__()

#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         augment_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#             transforms.RandomGrayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#         self.datasets = []

#         for i, metadata_value in enumerate(
#                 self.metadata_values(dataset, metadata_name)):
#             if augment and (i not in test_envs):
#                 env_transform = augment_transform
#             else:
#                 env_transform = transform

#             env_dataset = WILDSEnvironment(
#                 dataset, metadata_name, metadata_value, env_transform)

#             self.datasets.append(env_dataset)

#         self.input_shape = (3, 224, 224,)
#         self.num_classes = dataset.n_classes

#     def metadata_values(self, wilds_dataset, metadata_name):
#         metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
#         metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
#         return sorted(list(set(metadata_vals.view(-1).tolist())))


# class WILDSCamelyon(WILDSDataset):
#     ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
#             "hospital_4"]
#     def __init__(self, root, test_envs, hparams):
#         dataset = Camelyon17Dataset(root_dir=root)
#         super().__init__(
#             dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


# class WILDSFMoW(WILDSDataset):
#     ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
#             "region_4", "region_5"]
#     def __init__(self, root, test_envs, hparams):
#         dataset = FMoWDataset(root_dir=root)
#         super().__init__(
#             dataset, "region", test_envs, hparams['data_augmentation'], hparams)

# OLD method before confirming the 

# class MultipleEnvironmentImageFolder(MultipleDomainDataset):
#     def __init__(self, root, test_envs, augment, hparams):
#         super().__init__()
#         environments = [f.name for f in os.scandir(root) if f.is_dir() and "valid" in f.name]
#         #environments = [f.name for f in os.scandir(root) if f.is_dir()]
#         environments = sorted(environments) # list of all domains in the dataset, in sorted order
#         # environments = ["valid"]
#         # model = timm.create_model("vit_base_patch16_224.orig_in21k_ft_in1k",pretrained=True)
#         # data_config = timm.data.resolve_model_data_config(model)
#         # transform = timm.data.create_transform(**data_config, is_training=False)
#         MEAN = [0.485, 0.456, 0.406]
#         STD = [0.229, 0.224, 0.225]
#         # if hparams["backbone"]=="ViTBase":
#         #     MEAN = [0.5, 0.5, 0.5]
#         #     STD = [0.5, 0.5, 0.5]
#         # else:
#         #     MEAN = [0.485, 0.456, 0.406]
#         #     STD = [0.229, 0.224, 0.225]
#         #self.ENVIRONMENTS = environments
#         # transform = transforms.Compose([
#         #     #transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
#         #     transforms.ToTensor(),
#         #     transforms.Normalize(
#         #         mean=MEAN, std=STD)
#         # ])
        
        
        
        
        
        
#         transform = transforms.Compose([
#             transforms.RandomResizedCrop(size=224,interpolation=InterpolationMode.BILINEAR,antialias=True),
#             transforms.RandomHorizontalFlip(0.5),
#             transforms.ToTensor(),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize(
#                 mean=MEAN, std=STD)
#         ])

#         augment_transform = transforms.Compose([
#             # transforms.Resize((224,224)),
#             transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#             transforms.RandomGrayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         #[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]


#         self.datasets = []
#         for i, environment in enumerate(environments):

#             if hparams["eval"]:
#                 print("Eval transform")
#                 if 'mixed_same' in environments:
#                     env_transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor()])
#                 else:
#                     env_transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(
#                     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
#                     # env_transform = transforms.Compose([
#                     #                 transforms.Resize(size=256,interpolation=InterpolationMode.BILINEAR,antialias=True),
#                     #                 transforms.CenterCrop(224),
#                     #                 transforms.ToTensor(),
#                     #                 transforms.ConvertImageDtype(torch.float),
#                     #                 transforms.Normalize(
#                     #                     mean=MEAN, std=STD)
#                     #             ])
#             elif augment and (i not in test_envs):
#                 print('[INFO] Doing Data Augmentation')
#                 env_transform = augment_transform
#             elif 'train'==environment:
#                 print('[INFO] NOT Doing Special Data Augmentation')
#                 env_transform = transform
#             else:
#                 print('[INFO] Doing ImageNet validation Augmentation')
#                 env_transform = transforms.Compose([
#                     transforms.Resize(size=256,interpolation=InterpolationMode.BILINEAR,antialias=True),
#                     transforms.CenterCrop(224),
#                     transforms.ToTensor(),
#                     transforms.Normalize(
#                         mean=MEAN, std=STD)
#                 # if (hparams["backbone"]=="ViTBase") or (hparams["backbone"]=="DeiTBase"):
#                 #     env_transform = transforms.Compose([
#                 #         transforms.Resize(size=math.floor(224/0.9),interpolation=InterpolationMode.BICUBIC),
#                 #         transforms.CenterCrop(224),
#                 #         transforms.ToTensor(),
#                 #         transforms.Normalize(
#                 #             mean=MEAN, std=STD)
#                 #     ])
#                 # else:
#                 #     env_transform = transforms.Compose([
#                 #     transforms.Resize(size=256,interpolation=InterpolationMode.BILINEAR,antialias=True),
#                 #     transforms.CenterCrop(224),
#                 #     transforms.ToTensor(),
#                 #     transforms.Normalize(
#                 #         mean=MEAN, std=STD)
#             ])
                
            

#             if 'mixed_same' in environments:
#                 print("ImageNet-9 path used")
#                 path = os.path.join(root, environment,"val")
#             else:
#                 path = os.path.join(root, environment)
#             env_dataset = ImageFolder(path,
#                 transform=env_transform)
#             ################################ Code required for RCERM ################################ 
#             # env_dataset: <class 'torchvision.datasets.folder.ImageFolder'>, 
#             # Dataset ImageFolder
#             #     Number of datapoints: 2050
#             #     Root location: ../../DG/DomainBed/domainbed/data/PACS/art_painting
#             # Dataset ImageFolder
#             #     Number of datapoints: 2345
#             #     Root location: ../../DG/DomainBed/domainbed/data/PACS/cartoon
#             # Dataset ImageFolder
#             #     Number of datapoints: 1671
#             #     Root location: ../../DG/DomainBed/domainbed/data/PACS/photo
#             # Dataset ImageFolder
#             #     Number of datapoints: 3934
#             #     Root location: ../../DG/DomainBed/domainbed/data/PACS/sketch
#             # access it via:
#             ################################ Code required for RCERM ################################ 
#             self.datasets.append(env_dataset)
#             ################################ Code required for RCERM ################################ 
# #             # env_dataset containst first all elts of class 0, then class 1, ...class C-1
# #             for (batch_idx, sample_batched) in enumerate(self.datasets[i]):
# #             ###     if batch_idx==1:
# #             ###         break
# #                 print('im ',batch_idx,' :',sample_batched)
# #                 ## eg, im  0  : (<PIL.Image.Image image mode=RGB size=227x227 at 0x7FB44448C430>, 0)
#             ################################ Code required for RCERM ################################ 

            

#         self.input_shape = (3, 224, 224,)
#         self.num_classes = len(self.datasets[-1].classes)
#         #print("Classes ++++",self.datasets[-1].classes)