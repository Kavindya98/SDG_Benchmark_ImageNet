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
from datasets import *
from algorithm import *
import signal
import argparse
import time
import wandb
import numpy as np


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12300'
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

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    return train_sampler, train_loader, val_loader

def get_model(rank, args):
    #model = models.resnet50(weights='DEFAULT')
    model = ERM(args)
    # model = model.cuda(rank)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == "SGD":
        print("SGD Optimizer selected")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_scheduler == "StepLR":
        print("StepLR LR Scheduler selected")
        sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        sheduler = None
    return model, loss_fn, optimizer, sheduler


def validate(model, loss_fn, val_loader, rank):
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

def main_worker(rank, args):
    setup(rank, args.world_size)

    # TensorBoard logging
    
    wandb.login()

    # TensorBoard logging
    
    wandb.init(
    # Set the project where this run will be logged
    project=f"{args.log_path[8:]}", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_rank{rank}", 
    # Track hyperparameters and run metadata
    config=vars(args))

    #batch_size = 64
    train_sampler, train_loader, val_loader = get_dataloaders(rank, args)
    model, loss_fn, optimizer, scheduler = get_model(rank, args)
    total_steps = len(train_loader)
    checkpoint_freq = total_steps // 2 
    best_accuracy =0
    start_time = time.time()
    for epoch in range(args.max_epoch):
        train_sampler.set_epoch(epoch)
        #train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, rank, scaler)
        epoch_loss = 0
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            #print(f"Rank: {rank}, Epoch: {epoch}, Batch: {batch} Start",flush=True)
            X, y = X.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
            #print(f"Rank: {rank}, Epoch: {epoch}, Batch: {batch} Data to device",flush=True)
            optimizer.zero_grad()
           
            output = model(X)
            #print(f"Rank: {rank}, Epoch: {epoch}, Batch: {batch} Model Output",flush=True)
            loss = loss_fn(output, y)
            #print(f"Rank: {rank}, Epoch: {epoch}, Batch: {batch} Calculate Loss",flush=True)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            Avg_lr = np.mean([param_group['lr'] for param_group in optimizer.param_groups])
            #print(f"Rank: {rank}, Epoch: {epoch}, Batch: {batch} Middle",flush=True)
            total_steps_count = epoch*total_steps+(batch+1)
            wandb.log({'Train Loss':loss,
                       'Learning Rate':Avg_lr})
            if ((epoch)==0 and (batch)==0):
                
                val_loss, val_acc = validate(model, loss_fn, val_loader, rank)
                print(f"Epoch: {epoch}, Batch: {batch}/{len(train_loader)}, Train Loss: {epoch_loss/(batch+1)}, Val Loss: {val_loss}, Val Acc: {val_acc}",flush=True)
            
            if (batch+1)==total_steps:
                #print("Evaluvating",flush=True)
                val_loss, val_acc = validate(model, loss_fn, val_loader, rank)
                print(f"Epoch: {epoch}, Batch: {batch}/{len(train_loader)}, Train Loss: {epoch_loss/(batch+1)}, Val Loss: {val_loss}, Val Acc: {val_acc}",flush=True)
                
                wandb.log({'Loss/train': epoch_loss/(batch+1),
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
            #print("Left",flush=True)   
        
        if scheduler is not None:
            scheduler.step()
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
    parser.add_argument('--network', type=str, default="ResNet50")
    parser.add_argument('--dataset', type=str, default="ImageNet")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained', action='store_false')
    parser.add_argument('--Freeze_bn', action='store_true')
    parser.add_argument('--optimizer', type=str, default="SGD")
    parser.add_argument('--lr_scheduler', type=str, default=None)
    parser.add_argument('--log_path', type=str, default="Results/resnet_50_imagenet_correct")
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=5)
    args = parser.parse_args()

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
