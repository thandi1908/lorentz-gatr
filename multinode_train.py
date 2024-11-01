import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
from argparse import ArgumentParser
import h5py as h5
# import hdf5plugin
import math
import time
import random
from torch.utils.data import Dataset, DataLoader
from model import Wrapper
from torch_geometric.nn.aggr import MeanAggregation
import torch.nn.functional as F

import warnings
# warnings.filterwarnings("ignore")

# parallel training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import (
    DistributedSampler,
)  # Distribute data across multiple gpus
from torch.distributed import init_process_group, destroy_process_group

class JetDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file

        with h5.File(self.h5_file, 'r') as hf:
            self.length = hf['pid'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5.File(self.h5_file, 'r') as data:
                X = data["Pmu"][idx]
                Y = data["pid"][idx]
                W = data["weights"][idx]
                mask = np.all(np.abs(X) != 0, axis=1)
        
        return X.astype(np.float32), np.array(Y, dtype=np.float32), mask, W.astype(np.float32)


def sum_reduce(num, device):
    r''' Sum the tensor across the devices.
    '''
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def train_step(model, dataloader, cost, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    aggr = MeanAggregation()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch: {epoch}', mininterval=10)):
    # for batch_idx, batch in [[0,1]]:
        
        batch = [x.to(device) for x in batch]
        X, y, mask, weights = batch

        optimizer.zero_grad()  # Zero the gradients

        outputs = model(X)  # Forward pass
        outputs = outputs.squeeze(-1, -2)
        outputs = aggr(outputs, dim=-1)
        outputs = F.sigmoid(outputs)
        
        bce = cost(outputs.squeeze(), y) * weights
        
        loss = bce.mean()

        loss.backward()  # Backward pass
        
        #take care of unruly gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        
        optimizer.step()  # Update parameters

        running_loss += loss.item()


    dist.barrier()
    distributed_batch = sum_reduce(len(dataloader), device=device).item()
    distributed_loss = sum_reduce(running_loss, device=device).item()/distributed_batch

    return distributed_loss

def test_step(model, dataloader, cost, epoch, device):
    model.train()
    running_loss = 0.0
    aggr = MeanAggregation()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch: {epoch}", mininterval=10)):
        
        batch = [x.to(device) for x in batch]
        X, y, mask, weights = batch

        with torch.no_grad():

            outputs = model(X)  # Forward pass
            outputs = outputs.squeeze(-1, -2)
            outputs = aggr(outputs, dim=-1)
            outputs = F.sigmoid(outputs)
        
            bce = cost(outputs.squeeze(), y) * weights
            
            loss = bce.mean()

            running_loss += loss.item()

    dist.barrier()
    distributed_batch = sum_reduce(len(dataloader), device=device).item()
    distributed_loss = sum_reduce(running_loss, device=device).item()/distributed_batch

    return distributed_loss

def train_model(model, train_loader, test_loader, loss, optimizer, train_sampler, global_rank=0, num_epochs=100, device='cpu',patience=5, output_dir="", save_tag=""):
    model_save = f"best_model_{save_tag}.pt"

    losses = {
        "train_loss": [],
        "val_loss": [],
    }
   
    tracker = {
        "bestValLoss": np.inf,
        "bestEpoch": 0
    }

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        train_losses = train_step(model, train_loader, loss, optimizer, epoch, device)
        val_losses = test_step(model, test_loader, loss, epoch, device)
        
        losses["train_loss"].append(train_losses)

        losses["val_loss"].append(val_losses)

        if device == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {losses["train_loss"][-1]:.4f}, Val Loss: {losses["val_loss"][-1]:.4f}')

        if losses["val_loss"][-1] < tracker["bestValLoss"]:
                tracker["bestValLoss"] = losses["val_loss"][-1]
                tracker["bestEpoch"] = epoch
                
                dist.barrier()

                if global_rank==0:
                    torch.save(
                        model.module.state_dict(), f"{output_dir}/{model_save}"
                    )

        dist.barrier() # syncronise (top GPU is doing more work)

        # check the validation loss from each GPU:
        debug = False 
        if debug:
            print(f"Rank: {global_rank}, Device: {device}, Train Loss: {losses['train_loss'][-1]:.5f}, Validation Loss: {losses['val_loss'][-1]:.5f}")
            print(f"Rank: {global_rank}, Device: {device}, Best Loss: {tracker['bestValLoss']}, Best Epoch: {tracker['bestEpoch']}")
        # early stopping check
        if epoch - tracker["bestEpoch"] > patience:
            print(f"breaking on Rank: {global_rank}, device: {device}")
            break
        
    if global_rank==0:
        print(f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!")
    
        # save losses
        json.dump(losses, open(f"{output_dir}/training_{save_tag}.json", "w"))

def main(world_size, global_rank, rank, args, seed=25):
    # ddp_setup(rank, world_size)

    print(f"Training with random seed: {seed}")

    # set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


    # make dataset
    train_data = JetDataset(f"{args.data_dir}/train_atlas.h5")
    test_data = JetDataset(f"{args.data_dir}/valid_atlas.h5")

    sampler_train = DistributedSampler(train_data, shuffle=True, num_replicas=world_size, rank=global_rank)
    sampler_test = DistributedSampler(test_data, shuffle=False, num_replicas=world_size, rank=global_rank)

    # make dataloader
    train_loader = DataLoader(
            train_data, 
            batch_size=512*2,
            shuffle=False,
            pin_memory=True,
            sampler=sampler_train,
            num_workers=os.cpu_count()//world_size,
            # prefetch_factor=4
            )
    test_loader = DataLoader(
        test_data,
        batch_size=512*2,
        shuffle=False,
        pin_memory=True,
        sampler=sampler_test,
        num_workers=os.cpu_count()//world_size,
        # prefetch_factor=4

        )

    

    # set up model
    model = Wrapper()
    model = DDP(model.to(rank), device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    if rank==0:
        d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {d} Rank: {global_rank}")

    # train
    BCE = nn.BCELoss(reduction='none')
    train_model(
        model, 
        train_loader, 
        test_loader, 
        BCE, 
        optimizer, 
        device=rank,
        global_rank=global_rank,
        train_sampler=sampler_train,
        output_dir=args.outdir, 
        save_tag=args.save_tag
        )

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        default="",
        help="Directory of training and validation data"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="outdir",
        default="",
        help="Directory to output best model",
    )
    parser.add_argument(
        "--save_tag",
        dest="save_tag",
        default="",
        help="Extra tag for checkpoint model",
    )
    
    args = parser.parse_args()

    world_size = int(os.environ['WORLD_SIZE'])
    torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
    global_rank = torch.distributed.get_rank()
    rank = int(os.environ['LOCAL_RANK'])
    
    main(world_size, global_rank, rank, args)
