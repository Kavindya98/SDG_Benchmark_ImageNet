import os
from pathlib import Path
import re
from datasets import *
from algorithm import *
from imagenet_ddp import *
import argparse
import numpy as np
import json
import shutil
from tqdm import tqdm
import torch.nn.functional as F
from probabilities_to_decision import *
Mapping = ImageNetProbabilitiesTo16ClassesMapping()

def get_model_transforms(args):
    hparams = get_default_hparams(args)
    if hparams["backbone"]=="ViTBase":
        MEAN = [0.5, 0.5, 0.5]
        STD = [0.5, 0.5, 0.5]
    else:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]


    if (hparams["backbone"]=="ViTBase") or (hparams["backbone"]=="DeiTBase"):
        env_transform = transforms.Compose([
            transforms.Resize(size=math.floor(224/0.9),
                                interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN, std=STD)
        ])
    else:

        env_transform = transforms.Compose([
        transforms.Resize(size=256,
                    interpolation=InterpolationMode.BILINEAR,antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN, std=STD) 
    ])
        
    # model = ERM(args)
    # model.eval()
    # return model, env_transform
    return env_transform

def cue_conflicts_conversion(output):
    for j in range(len(output)):
        result=Mapping.probabilities_to_decision(output[j].cpu().squeeze().numpy())
    return result

def main_process(args_list):
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--world_size', type=int, default=5)
    parser.add_argument('--network', type=str, default="ViTBase")
    parser.add_argument('--dataset', type=str, default="ImageNet")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained', action='store_false')
    parser.add_argument('--Freeze_bn', action='store_true')
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--lr_scheduler', type=str, default=None)
    parser.add_argument('--log_path', type=str, default="Results/resnet_50_imagenet_correct")
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--loss_aug', action='store_false')
    parser.add_argument('--alpha_min', type=float, default=0.0)
    parser.add_argument('--consistancy_loss', type=float, default=12.0)
    parser.add_argument('--kernel_size', '-ks', type=int, default=[1,3,5,7], nargs='+',
                        help='kernal size for random layer, could be multiple kernels for multiscale mode')
    parser.add_argument('--mixing', '-mix', action='store_false',
                        help='mix the output of rand conv layer with the original input')
    parser.add_argument('--clamp_output', '-clamp', action='store_false',
                        help='clamp value range of randconv outputs to a range (as in original image)'
                        )
    parser.add_argument('--head_re_initialized', action='store_true')
    parser.add_argument('--identity_prob', '-idp', type=float, default=0.0,
                        help='the probability that the rand conv is a identity map, '
                                'in this case, the output and input must have the same channel number')
    parser.add_argument('--mixture_width', type=int, default=3)
    parser.add_argument('--mixture_depth', type=int, default=-1)
    parser.add_argument('--aug_severity', type=float, default=3.0)
    parser.add_argument('--all_ops', action='store_true')
    parser.add_argument('--pretrained_custom_weights', action='store_true')
    parser.add_argument('--lr_adv', '-lr_adv', default=0.00001, type=float,
        help='learning rate')
    parser.add_argument('--pre_epoch', type=int, default=4,
        help='number of epochs to pretrain on the source domains')
    parser.add_argument('--elbo_beta', '-beta', default=0.1, type=float)
    parser.add_argument('--clw', type=float, default=0.75,
        help='weight for invariant loss')
    parser.add_argument('--num_blocks', type=int, default=4,
        help='get outputs from a random intermediate layer of transnet')
    parser.add_argument('--adv_steps', '-nadv', default=10, type=int)

    args = parser.parse_args(args_list.split())
    #print(args)

    args.head_re_initialized=False

    loss_fn = nn.CrossEntropyLoss()
    #root = "/home/kavindya/data/Model/ImageNet_training/Results/RandConv_CNN_with_ranConv_validation_alpha_0.5"
    root = args.log_path
    model_paths = [ i.name for i in os.scandir(root) if "checkpoint_best" in i.name]

    print(args.log_path[8:],flush=True)

    base_path = Path("/media/SSD2/Dataset/Cue_conflicts_stimuli/texture")
    clx = {l:i for i,l in enumerate(sorted(os.listdir("/media/SSD2/Dataset/Cue_conflicts_stimuli/texture")))}


    model_array= {}
    for i in model_paths:
        model_array[i] = ERM(args)
        saved_state_dict = torch.load(os.path.join(root,i))
        model_array[i].load_state_dict(saved_state_dict["state_dict"])
        model_array[i].eval()
    
    env_transform = get_model_transforms(args)

    total_shape_bias = []
    total_texture_bias = []

    for i in model_array:
        model = model_array[i]        
        shape_acc, texture_acc = 0, 0
        mis_classified = [] 
    
        for i in base_path.rglob('*.png'):
            #classes= [re.findall(r'[a-zA-Z]+', k)[0] for k in str(i).split("/")[-1][:-4].split("-")]
            shape = str(i).split("/")[-2]
            texture = re.findall(r'[a-zA-Z]+', str(i).split("/")[-1][:-4].split("-")[-1])[0]
            classes = [shape,texture]
            if classes[0]!=classes[1]:
                with torch.no_grad():
                    img = Image.open(str(i)).convert('RGB')
                    img = env_transform(img)
                    img = img.unsqueeze(0)
                    output = model(img)
                    output = F.softmax(output,dim=1)
                    pred = cue_conflicts_conversion(output)
                    if pred == classes[0]:
                        shape_acc+=1
                    elif pred == classes[1]:
                        texture_acc+=1
                    else:
                        mis_classified.append([pred,classes])
        shape_bias = round((shape_acc/(shape_acc+texture_acc))*100,2)
        texture_bias = round((texture_acc/(shape_acc+texture_acc))*100,2)

        print("Shape Accuracy: ",shape_acc," Texture Accuracy: ",texture_acc,flush=True)
        print("Shape_bias: ",shape_bias," Texture_bias: ",texture_bias,flush=True)
        print(flush=True)

        total_shape_bias.append(shape_bias)
        total_texture_bias.append(texture_bias)
    
    print("Overall shape bias: ", np.mean(total_shape_bias), " Overall texture bias: ",np.mean(total_texture_bias),flush=True)
    print("*******************************************************************************",flush=True)
    print(flush=True)


def main():
    # network_list = ["--loss_aug --max_epoch 30 --Freeze_bn --optimizer SGD --network ResNet50  --log_path Results/RandConv_CNN_Head_ReInitialized_with_ranConv_validation_different_ERM --lr 0.0001 --wd 0 --world_size 4 --batch_size 64 --pretrained --pretrained_custom_weights --head_re_initialized",
    # "--max_epoch 30 --Freeze_bn --optimizer SGD --network ResNet50  --log_path Results/ResNet50_AugMix_CNN_Head_ReInitialized_different_ERM --lr 0.0001 --wd 0 --world_size 4 --batch_size 64 --pretrained --pretrained_custom_weights --head_re_initialized",
    # "--max_epoch 30 --lr_adv 0.00005 --Freeze_bn --optimizer SGD --elbo_bet 0.1 --lr_scheduler CosineLR --num_blocks 0 --pre_epoch 4 --network ResNet50  --log_path Results/ABA_CNN_Head_ReInitialized_different_ERM_3 --lr 0.004 --wd 0 --world_size 4 --adv_steps 10 --batch_size 64 --pretrained --pretrained_custom_weights --head_re_initialized",
    # "--loss_aug --consistancy_loss 10 --max_epoch 30 --optimizer Adam --network DeiTSmall  --log_path Results/RandConv_DeiTSmall_Head_ReInitialized_with_ranConv_validation --wd 0 --world_size 4 --batch_size 64 --pretrained --head_re_initialized",
    # "--max_epoch 30 --optimizer Adam --network DeiTSmall  --log_path Results/DeiTSmall_AugMix_CNN_Head_ReInitialized --wd 0 --world_size 4 --batch_size 64 --pretrained --head_re_initialized",
    # "--max_epoch 30 --lr_adv 0.0005 --optimizer Adam --elbo_bet 1 --num_blocks 0 --pre_epoch 4 --network DeiTSmall --log_path Results/ABA_DeiTSmall_Head_ReInitialized --wd 0 --world_size 4 --adv_steps 10 --batch_size 64 --pretrained --head_re_initialized",
    # "--loss_aug --consistancy_loss 10 --max_epoch 30 --optimizer Adam --network DeiTBase  --log_path Results/RandConv_DeiTBase_Head_ReInitialized_with_ranConv_validation --wd 0 --world_size 4 --batch_size 64 --pretrained --head_re_initialized",
    # "--max_epoch 30 --optimizer Adam --network DeiTBase  --log_path Results/DeiTBase_AugMix_CNN_Head_ReInitialized_5 --wd 0 --world_size 4 --batch_size 64 --pretrained --head_re_initialized",
    # "--max_epoch 30 --lr_adv 0.0005 --optimizer Adam --elbo_bet 1 --num_blocks 0 --pre_epoch 4 --network DeiTBase --log_path Results/ABA_DeiTBase_Head_ReInitialized --wd 0 --world_size 4 --adv_steps 10 --batch_size 64 --pretrained --head_re_initialized",
    # "--loss_aug --consistancy_loss 10 --max_epoch 30 --optimizer Adam --network ViTBase  --log_path Results/RandConv_ViTBase_Head_ReInitialized_with_ranConv_validation --wd 0 --world_size 4 --batch_size 64 --pretrained --head_re_initialized",
    # "--max_epoch 30 --optimizer Adam --network ViTBase --log_path Results/ViTBase_AugMix_CNN_Head_ReInitialized_2 --wd 0 --world_size 4 --batch_size 64 --pretrained --head_re_initialized",
    # "--max_epoch 30 --lr_adv 0.0005 --optimizer Adam --elbo_bet 1 --num_blocks 0 --pre_epoch 4 --network ViTBase --log_path Results/ABA_ViTBase_Head_ReInitialized --wd 0 --world_size 4 --adv_steps 10 --batch_size 64 --pretrained --head_re_initialized",

    # ]

    network_list =  ["--max_epoch 30 --lr_adv 0.0005 --optimizer Adam --elbo_bet 1 --num_blocks 0 --pre_epoch 4 --network DeiTSmall --log_path Results/ABA_DeiTSmall_Head_ReInitialized_2 --wd 0 --world_size 4 --adv_steps 10 --batch_size 64 --pretrained --head_re_initialized",
    "--loss_aug --consistancy_loss 10 --max_epoch 30 --optimizer Adam --network ViTBase  --log_path Results/RandConv_ViTBase_Head_ReInitialized_with_ranConv_validation --wd 0 --world_size 4 --batch_size 64 --pretrained --head_re_initialized",
    "--max_epoch 30 --optimizer Adam --network ViTBase --log_path Results/ViTBase_AugMix_CNN_Head_ReInitialized_2 --wd 0 --world_size 4 --batch_size 64 --pretrained --head_re_initialized",
    "--max_epoch 30 --lr_adv 0.0005 --optimizer Adam --elbo_bet 1 --num_blocks 0 --pre_epoch 4 --network ViTBase --log_path Results/ABA_ViTBase_Head_ReInitialized --wd 0 --world_size 4 --adv_steps 10 --batch_size 64 --pretrained --head_re_initialized",

    ]

    for args_list in network_list:
        main_process(args_list)




if __name__ == "__main__":
    main()
