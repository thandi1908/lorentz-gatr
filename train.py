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

def train_step(model, dataloader, cost, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    aggr = MeanAggregation()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch: {epoch}", mininterval=10)):
        
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


    running_loss /=len(dataloader)

    return running_loss

def test_step(model, dataloader, cost, optimizer, epoch, device):
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

    running_loss /=len(dataloader)

    return running_loss

def train_model(model, train_loader, test_loader, loss, optimizer, num_epochs=100, device='cpu',patience=5, output_dir="", save_tag=""):

    losses = {
        "train_loss": [],
        "val_loss": [],
    }
   
    tracker = {
        "bestValLoss": np.inf,
        "bestEpoch": 0
    }

    model.to(device)

    for epoch in range(num_epochs):
        train_losses = train_step(model, train_loader, loss, optimizer, epoch, device)
        val_losses = test_step(model, test_loader, loss, epoch, device)
        
        losses["train_loss"].append(train_losses)

        losses["val_loss"].append(val_losses)

        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {losses["train_loss"][-1]:.4f}, Val Loss: {losses["val_loss"][-1]:.4f} ')

        if losses["val_loss"][-1] < tracker["bestValLoss"]:
                tracker["bestValLoss"] = losses["val_loss"][-1]
                tracker["bestEpoch"] = epoch
                
                torch.save(
                    model.module.state_dict(), f"{output_dir}/{model_save}"
                )

        # early stopping check
        if epoch - tracker["bestEpoch"] > patience:
            break
        
  
        print(f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!")
    
        # save losses
        json.dump(losses, open(f"{output_dir}/training_{save_tag}.json", "w"))

def main(args, seed=25):
    # ddp_setup(rank, world_size)

    print(f"Training with random seed: {seed}")

    # set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


    # make dataset
    train_data = JetDataset(f"{args.data_dir}/train_atlas.h5")
    test_data = JetDataset(f"{args.data_dir}/valid_atlas.h5")

    # make dataloader
    train_loader = DataLoader(
            train_data, 
            batch_size=256,
            shuffle=True,
            pin_memory=True,
            # prefetch_factor=4
            )
    test_loader = DataLoader(
        test_data,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        # prefetch_factor=4

        )

    

    # set up model
    model = Wrapper()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {d}")

    # train
    BCE = nn.BCELoss(reduction='none')
    train_model(
        model, 
        train_loader, 
        test_loader, 
        BCE, 
        optimizer, 
        device=d,
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

    print("Doing something....")
    
    main(args)
