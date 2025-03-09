import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler
from ABA.multi_bnn import Multi_BNN
from datasets import *
from algorithm import *
import signal
import argparse
from rand_conv import *
import time
#from torch.utils.tensorboard import SummaryWriter
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12305'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Register signal handler for termination signals
    signal.signal(signal.SIGINT, signal_handler)  # Register Ctrl+C signal
    signal.signal(signal.SIGTERM, signal_handler)  # Register termination signal


def cleanup():
    dist.destroy_process_group()

# Signal handler function
def signal_handler(signal, frame):
    print("Received termination signal. Cleaning up...")
    cleanup()
    exit(0)

def get_default_hparams(args):

    hparams = {}
    hparams['data_augmentation']=False
    hparams["backbone"]= args.network
    hparams["mean_std"]=[[0.5]*3,[0.5*3]]
    hparams["normalization"]=True

    return hparams

def get_dataloaders(rank, args):

    hparams = get_default_hparams(args)

    data = ImageNet("/media/SSD2/Dataset",[1],hparams)
    train_dataset, val_dataset = data[0],data[1]
    # train_dataset.samples = train_dataset.samples[:int(len(train_dataset.samples)*0.5)]
    # train_dataset.targets = train_dataset.targets[:int(len(train_dataset.targets)*0.5)]

    print("Mean and STD ", hparams["mean_std"])

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    return train_sampler, train_loader, val_loader, hparams

def get_model(rank, args, max_iter):
    #model = models.resnet50(weights='DEFAULT')
    model = ERM(args)
    if args.pretrained_custom_weights:
        saved_state_dict = torch.load("/home/kavindya/data/Model/ImageNet_training/Results/ResNet50_CNN_From_Scratch/checkpoint_best_rank3.pkl")
        model.load_state_dict(saved_state_dict["state_dict"])
        print("Got Our ResNet ERM pretrained weights",flush=True)

    if args.head_re_initialized:
        model = Re_Initialize_Head(args,model)
    
    # model = model.cuda(rank)
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == "SGD":
        print("SGD Optimizer selected")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == "Adam":
        print("Adam Optimizer selected") # it was previously mention SGD as a mistake
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.lr_scheduler == "StepLR":
        print("StepLR LR Scheduler selected")
        sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.lr_scheduler == "CosineLR":
        print("CosineLR LR Scheduler selected")
        sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)

    else:
        sheduler = None
    scaler = GradScaler()
    return model, loss_fn, optimizer, sheduler, scaler



def validate_with_RandConv(model,rand_module, loss_fn, val_loader, rank):
    model.eval()
    val_loss = 0
    correct = 0
    total=0
    with torch.no_grad():
        for X, y in val_loader:
            #X, y = X.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
            X, y = X.to(rank), y.to(rank)
            rand_module.randomize()
            output = model(rand_module(X))
            val_loss += loss_fn(output, y).item()*y.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            total += y.size(0)
            correct += pred.eq(y.view_as(pred)).sum().item()


    model.train()
    # if rank == 0:
    #     print(f"Validation Loss: {val_loss}, Accuracy: {100. * correct / len(val_loader.dataset)}%")
    return (val_loss / total),(100. * correct / total)

def validate(model,loss_fn, val_loader, rank):
    model.eval()
    val_loss = 0
    correct = 0
    total=0
    with torch.no_grad():
        for X, y in val_loader:
            #X, y = X.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
            X, y = X.to(rank), y.to(rank)
            output = model(X)
            val_loss += loss_fn(output, y).item()*y.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            total += y.size(0)
            correct += pred.eq(y.view_as(pred)).sum().item()

    model.train()
    # if rank == 0:
    #     print(f"Validation Loss: {val_loss}, Accuracy: {100. * correct / len(val_loader.dataset)}%")
    return (val_loss / total),(100. * correct / total)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves model and training parameters at checkpoint. Only the process with rank 0 saves the checkpoint.
    Args:
        state: Contains model's state_dict, may contain other keys such as epoch, optimizer state.
        filename: Name of the checkpoint file.
    """
    torch.save(state, filename)

def get_random_module(args, data_mean, data_std):
    return RandConvModule(net=None,
                          in_channels=3,
                          out_channels=3,
                          kernel_size=args.kernel_size,
                          mixing=args.mixing,
                          identity_prob=args.identity_prob,
                          rand_bias=False,
                          distribution='kaiming_normal',
                          data_mean=data_mean,
                          data_std=data_std,
                          clamp_output=args.clamp_output,
                          alpha_min=args.alpha_min
                          )
def inv_loss_compute(model, rand_module, X, output):
    rand_module.randomize()
    output1 = model(rand_module(X))
    print("[INFO] Working upto 2nd randconv augmentation")

    rand_module.randomize()
    output2 = model(rand_module(X))
    print("[INFO] Working upto 3rd randconv augmentation")

    p_clean, p_aug1, p_aug2 = F.softmax(
                        output, dim=1), F.softmax(
                        output1, dim=1), F.softmax(
                        output2, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    inv_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    
def get_BCNN_module(args, data_mean, data_std, rank):
    return Multi_BNN(
        out_channels=3,
        kernel_size=args.kernel_size,
        mixing=args.mixing,
        clamp=args.clamp_output,
        num_blocks=args.num_blocks,
        data_mean=data_mean,
        data_std=data_std,
        device=rank
    )

def main_worker(rank, args):
    
    setup(rank, args.world_size)
    wandb.login()

    # TensorBoard logging
    #rank = rank+2
    wandb.init(
    # Set the project where this run will be logged
    project=f"{args.log_path[8:]}", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_rank{rank}", 
    # Track hyperparameters and run metadata
    config=vars(args))

    #batch_size = 64
    train_sampler, train_loader, val_loader, hparams = get_dataloaders(rank, args)
    max_iteration = len(train_loader)*args.max_epoch
    model, loss_fn, optimizer, scheduler, scaler = get_model(rank, args, max_iteration)
    # rand_module = get_random_module(args,data_mean=hparams["mean_std"][0], data_std=hparams["mean_std"][1])
    # rand_module = rand_module.to(rank)
    bcnn_module = get_BCNN_module(args,hparams["mean_std"][0],hparams["mean_std"][1],rank)
    bcnn_module.to(rank)
    aug_optimizer = torch.optim.Adam(bcnn_module.parameters(),args.lr_adv)
    total_steps = len(train_loader)

    best_accuracy =0
    start_time = time.time()
    inv_loss, previous_loss = 0, 0
    eps = 1e-8
    for epoch in range(args.max_epoch):
        train_sampler.set_epoch(epoch)
        #train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, rank, scaler)
        epoch_loss = 0
        epoch_inv_loss =0
        epoch_org_loss =0
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            
            with autocast():
                try:
                    # Assuming 'rank' is defined and is the index of the GPU for the current process
                    X, y = X.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
                    
                    # model.to(device)  # Ensure model is on the right device
                    # bcnn_module.to(device)  # Ensure bcnn_module is on the right device
                

                    if epoch >= args.pre_epoch:
                        
                        model.eval()
                        bcnn_module.randomize()
                        bcnn_module.train()

                        for n in range(args.adv_steps):
                            
                            model.zero_grad()
                            aug_optimizer.zero_grad()
                            bcnn_module.zero_grad()
                            #try:
                            X_aug_1, bnn_kl = bcnn_module(X)
                            
                            y_g = model(X_aug_1)
                            # except Exception as e:
                            #     print(f"Error in augmentation step {n}: {str(e)}",flush=True)
                            
                            loss_aug = -loss_fn(y_g, y) - args.elbo_beta * bnn_kl
                            
                            loss_aug.backward()
                            aug_optimizer.step()
                            

                        X_aug_1, _ = bcnn_module(X)
                        X_aug_2, _ = bcnn_module(X)

                        X_concat = torch.cat((X, X_aug_1, X_aug_2), dim=0)
                        output, output1, output2 = model(X_concat).chunk(3, dim=0)

                        p_clean, p_aug1, p_aug2 = F.softmax(
                                            output, dim=1), F.softmax(
                                            output1, dim=1), F.softmax(
                                            output2, dim=1)
                        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                        inv_loss = (F.kl_div(p_mixture, p_clean+eps, reduction='batchmean') +
                                    F.kl_div(p_mixture, p_aug1+eps, reduction='batchmean') +
                                    F.kl_div(p_mixture, p_aug2+eps, reduction='batchmean')) / 3.
                        
                        model.train()
                        #bcnn_module.eval()
                        ce_loss = loss_fn(output, y)
                        loss =  (1-args.clw)*ce_loss + args.clw*inv_loss
                    
                    else:

                        output = model(X)
                        ce_loss = loss_fn(output, y)
                        inv_loss = torch.zeros(1)
                        loss = ce_loss

                except Exception as e:
                    print(f"An error occurred during processing: {str(e)}")
                

            epoch_inv_loss+=inv_loss.item()
            
            #previous_loss = loss.item()
            model.zero_grad()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if args.lr_scheduler == "CosineLR" :
                scheduler.step()

            epoch_org_loss+=ce_loss.item()
            Avg_lr = np.mean([param_group['lr'] for param_group in optimizer.param_groups])
            epoch_loss += loss.item()
        
            wandb.log({'Train Loss/Inv_loss': inv_loss.item(),
                    'Train Loss/Before_inv': ce_loss.item(),
                    'Train Loss/After_inv':loss.item(),
                    'Learning Rate':Avg_lr})

            if ((epoch)==0 and (batch)==0) :
                
                val_loss, val_acc = validate(model, loss_fn, val_loader, rank)
                print(f"Epoch: {epoch}, Batch: {batch}/{len(train_loader)}, Train Loss: {epoch_loss/(batch+1)}, Val Loss: {val_loss}, Val Acc: {val_acc}",flush=True)
              
            if (batch+1)==total_steps:
                
                val_loss, val_acc = validate(model, loss_fn, val_loader, rank)
                print(f"Epoch: {epoch}, Batch: {batch}/{len(train_loader)}, Train Loss: {epoch_loss/(batch+1)}, Val Loss: {val_loss}, Val Acc: {val_acc}",flush=True)
                
                wandb.log({'Loss/train': epoch_loss/(batch+1),
                        'Loss/train_inv': epoch_inv_loss/(batch+1),
                        'Loss/train_org': epoch_org_loss/(batch+1),
                        'Loss/validation':val_loss,
                        'Accuracy/validation': val_acc})
                if best_accuracy<val_acc:
                    best_accuracy=val_acc
                    with open(f'{args.log_path}/validation_accuracy_rank{rank}.txt', 'a') as f:
                        f.write(f'Best Model upto Now\n')
                        f.write(f'Epoch: {epoch}, Batch: {batch}/{len(train_loader)}, Train Loss: {epoch_loss/(batch+1)}, Val Loss: {val_loss}, Val Acc: {val_acc}\n')
                    save_checkpoint({
                            'epoch': epoch,
                            'state_dict': model.module.state_dict(),  # Note: Use model.module.state_dict() to save the model state
                            'optimizer': optimizer.state_dict(),
                            # Add other items you want to save, e.g., scheduler state
                        }, filename=os.path.join(args.log_path, f"checkpoint_best_rank{rank}.pkl"))
                else:
                     with open(f'{args.log_path}/validation_accuracy_rank{rank}.txt', 'a') as f:
                        f.write(f'Epoch: {epoch}, Batch: {batch}/{len(train_loader)}, Train Loss: {epoch_loss/(batch+1)}, Val Loss: {val_loss}, Val Acc: {val_acc}\n')
            #del output, output1, output2, X_aug_1, X_aug_2, X_concat, p_clean, p_aug1, p_aug2, p_mixture, inv_loss, ce_loss, loss
            torch.cuda.empty_cache()
            #print("Left",flush=True)   

        if scheduler is not None and args.lr_scheduler != "CosineLR":
            scheduler.step()

        torch.cuda.empty_cache()
        save_checkpoint({'epoch': epoch,
                        'state_dict': model.module.state_dict(),  # Note: Use model.module.state_dict() to save the model state
                        'optimizer': optimizer.state_dict(),
                            # Add other items you want to save, e.g., scheduler state
                        }, filename=os.path.join(args.log_path, f"checkpoint_last_rank{rank}.pkl"))

        # if rank == 0:  # Check if the process is the main process
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.module.state_dict(),  # Note: Use model.module.state_dict() to save the model state
        #         'optimizer': optimizer.state_dict(),
        #         # Add other items you want to save, e.g., scheduler state
        #     }, filename=f"checkpoints/imagenet/checkpoint_epoch_{epoch + 1}.pth.tar")
        # validate(model, loss_fn, val_loader, rank)
    
    # End time
    end_time = time.time()

    # Calculate the duration in hours
    duration_hours = (end_time - start_time) / 3600
    print(f"Training completed in {duration_hours:.2f} hours +++++++++++++++")

    # Close TensorBoard writer
    wandb.finish()

    cleanup()

def main():
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
    parser.add_argument('--head_re_initialized', action='store_true')

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

    parser.add_argument('--alpha_min', type=float, default=0.0)
    parser.add_argument('--consistancy_loss', type=float, default=10.0)
    parser.add_argument('--kernel_size', '-ks', type=int, default=[1,3,5,7], nargs='+',
                        help='kernal size for random layer, could be multiple kernels for multiscale mode')
    parser.add_argument('--mixing', '-mix', action='store_true',
                        help='mix the output of rand conv layer with the original input')
    parser.add_argument('--clamp_output', '-clamp', action='store_false',
                        help='clamp value range of randconv outputs to a range (as in original image)'
                        )
    parser.add_argument('--identity_prob', '-idp', type=float, default=0.0,
                        help='the probability that the rand conv is a identity map, '
                             'in this case, the output and input must have the same channel number')
    parser.add_argument('--pretrained_custom_weights', action='store_true')


    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    print(args)
    os.makedirs(args.log_path, exist_ok=True)

    try:
        mp.spawn(main_worker, args=(args,), nprocs=args.world_size)#, join=True)
    except KeyboardInterrupt:
        print("[INFO] Interrupted")
        try:
            dist.destroy_process_group()
        except KeyboardInterrupt:
            os.system("kill $(ps aux | grep multipreocessing.spawn | grep -v grep | awk '{print $2})")

if __name__ == "__main__":
    main()
