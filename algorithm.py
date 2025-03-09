import torch.nn as nn
import torchvision
import torch
import random
import timm
#import pretrained_models.DeiT_models.visiontransformer as visiontransformer
import os
from lib.t2t_vit import tfsvit_t2t_vit_t_14, atfsvit_t2t_vit_t_14
from lib.t2t_vit import t2t_vit_t_14
from lib.t2t_utils import load_for_transfer_learning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_network(args):
    if args.network =="ResNet50":
        model = torchvision.models.resnet50(pretrained=True)
        print("Using Pretrained ResNet50",flush=True)
        
    elif args.network == "ViTBase":
        model = timm.create_model("vit_base_patch16_224.orig_in21k_ft_in1k",pretrained=True)
        print("Using Pretrained ViTBase",flush=True)
        
    elif args.network == "DeiTBase":
        model = timm.create_model("deit_base_patch16_224.fb_in1k",pretrained=True)
        print("Using Pretrained DeiTBase",flush=True)
        
    elif args.network == "T2T14":
        model = t2t_vit_t_14()
        # load the pretrained weights
        pretrained_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained_models', 't2t',
                                       '81.7_T2T_ViTt_14.pth.tar')
        load_for_transfer_learning(model, pretrained_path, use_ema=True, strict=True, num_classes=1000)
        print("Using Pretrained T2T14",flush=True)
        
    elif args.network == "DeiTSmall":
        model = torch.hub.load(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained_models', 'DeiT_models'),
            'deit_small_patch16_224',
            pretrained=True, source='local')
        print("Using Pretrained DeiTSmall",flush=True)
       
    else:
        model = None
        print("Unknown model",flush=True)
    return model

def Re_Initialize_Head(args,model):
    if args.network =="ResNet50":
        num_features = model.model.fc.in_features
        model.model.fc = nn.Linear(num_features, 1000)
        print("Head Re-Initialized - ResNet50",flush=True)
                
    elif args.network == "ViTBase":
        num_features = model.model.head.in_features
        model.model.head = nn.Linear(num_features, 1000)
        print("Head Re-Initialized - ViTBase",flush=True)
        
    elif args.network == "DeiTBase":
        num_features = model.model.head.in_features
        model.model.head = nn.Linear(num_features, 1000)
        print("Head Re-Initialized - DeiTBase",flush=True)

    elif args.network == "T2T14":
        model.model.head = nn.Linear(384, 1000)
        print("Head Re-Initialized - T2T14",flush=True)

    elif args.network == "DeiTSmall":
        model.model.head = nn.Linear(384, 1000)
        print("Head Re-Initialized - DeiTSmall",flush=True)

    else:
        model = None
        print("Unknown model",flush=True)
    return model


class ERM(nn.Module):
    def __init__(self, args):
        super(ERM, self).__init__()
        self.args = args
        self.model = get_network(args)
        if self.args.Freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        out = self.model(x)
        if self.args.network =="ResNet50" or self.args.network =="DeiTBase" or self.args.network =="ViTBase":
            return out
        else:
            return out[-1]
    
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.args.Freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

