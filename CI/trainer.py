from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from lightning.pytorch import seed_everything
from os import listdir
from os.path import isfile, join
import xarray as xr
import numpy as np
import pandas as pd
import joblib
import argparse
import yaml
from model import CI_unet
from sklearn.model_selection import GroupShuffleSplit

def get_convective_day(ts):
    
    ts = pd.Timestamp(ts)
    
    if ts.hour < 12:
        ts = ts - pd.Timedelta(days=1)
        
    return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day, hour=12)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file")
args = parser.parse_args()
with open(args.config) as config_file:
    config = yaml.safe_load(config_file)

seed_everything(config["seed"])
   
ds_labels = xr.open_dataset(config["labels_path"]+'all_train_labels_2022_2023.nc')

all_train_labels = ds_labels.labels.values
all_train_times = ds_labels.time.values

print('10min', flush=True)

## load all training data (10min)
all_train_preds = np.load(config["predictors_path"]+"training_data_full_10min_2022_2023.npy")

## split/shuffle training and validation datasets
convective_day_splits = pd.Series(all_train_times).apply(get_convective_day)

# Initialize GroupShuffleSplit with 1 split
gss = GroupShuffleSplit(n_splits=1, train_size=config["train_size"], random_state=config["seed"])

# get the split
train_idxs, val_idxs = next(gss.split(all_train_times, groups=convective_day_splits))

# filter the numpy arrays using the generated split                          
train_data = all_train_preds[train_idxs,:,:,:]
train_label = all_train_labels[train_idxs,:,:]
val_data = all_train_preds[val_idxs,:,:,:]
val_label = all_train_labels[val_idxs,:,:]

print(train_data.shape, flush=True)
print(val_data.shape, flush=True)

# Find samples (along spatial axis) that contain NaNs
has_nan_train = np.isnan(train_data).any(axis=(2, 3))
has_nan_val = np.isnan(val_data).any(axis=(2, 3))

# Indices of samples with NaNs (train)
nan_indices_train = np.where(has_nan_train)[0]
nan_indices_train = np.unique(nan_indices_train)

# Indices of samples with NaNs (val)
nan_indices_val = np.where(has_nan_val)[0]
nan_indices_val = np.unique(nan_indices_val)

## delete any of the NaN samples from train/val
train_data = np.delete(train_data, nan_indices_train, axis=0)
train_label = np.delete(train_label, nan_indices_train, axis=0)
val_data = np.delete(val_data, nan_indices_val, axis=0)
val_label = np.delete(val_label, nan_indices_val, axis=0)

## transpose train/val arrays for scaling                          
train_data = np.transpose(train_data, (0, 2, 3, 1))
val_data = np.transpose(val_data, (0, 2, 3, 1))  

## get scaler from config
if config["scaler_type"] == 'standard':                         
    scaler = StandardScaler()

## scale data                          
train_data_scaled = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
val_data_scaled = scaler.transform(val_data.reshape(-1, val_data.shape[-1])).reshape(val_data.shape) 

#joblib.dump(scaler, '/storage/group/sjg213/default/cps5892/U-Net/scaler_UNet_10min_2022_2023.bin', compress=True)

## transpose back to proper dimension order                          
train_data_scaled = np.transpose(train_data_scaled, (0, 3, 1, 2))
val_data_scaled = np.transpose(val_data_scaled, (0, 3, 1, 2))

## make numpy arrays pytorch tensors                          
train_data_scaled = torch.tensor(train_data_scaled, dtype=torch.float32)
train_label = torch.tensor(train_label, dtype=torch.long)                          
val_data_scaled = torch.tensor(val_data_scaled, dtype=torch.float32)
val_label = torch.tensor(val_label, dtype=torch.long)

## recognize GPU                          
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## get CI_unet model from config parameters
model = CI_unet(**config["model"])
model.to(device)

## get weights                          
weights = torch.tensor(config["model"]["loss_weights"])

## initialize loss function (weighted cross entropy)                          
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

## initialize optimizer (AdamW)                          
optimizer = torch.optim.AdamW(model.parameters(), lr=config["model"]["lr"])
                          
#scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult = 1, eta_min = 1e-6) 

# LR schedule                          
scheduler = ReduceLROnPlateau(
    optimizer,
    mode=config["callbacks"]["ReduceLROnPlateau"]["mode"],         # minimize val loss
    factor=config["callbacks"]["ReduceLROnPlateau"]["factor"],         # reduce LR by this factor
    patience=config["callbacks"]["ReduceLROnPlateau"]["patience"],         # wait this many epochs before reducing
)

# create pytorch tensor datasets, data loaders, and read in batch size for training                          
batch_size = config["model"]["batch_size"]
train_dataset = TensorDataset(train_data_scaled, train_label)
val_dataset = TensorDataset(val_data_scaled, val_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# set number of epochs                          
num_epochs = config["model"]["epochs"] 
best_val_loss = float("inf")

# ==== Training Loop ====                        
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    # Wrap train_loader with tqdm for a progress bar
    train_loader_tqdm = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}")
    
    for batch_X, batch_y in train_loader_tqdm:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)  # output shape: (B, 4, 128, 128)

        loss = criterion(logits, batch_y)  # batch_y: (B, 128, 128)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        # Apply softmax to get class probabilities in inference
        probs = torch.softmax(logits, dim=1)  # shape: [B, C, H, W]

        # Ground truth class labels should also be [B, H, W]
        targets = batch_y.long()

        train_loader_tqdm.set_postfix(loss=epoch_loss / (train_loader_tqdm.n + 1))

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0

    val_loader_tqdm = tqdm(val_loader, desc=f"[Val]   Epoch {epoch+1}/{num_epochs}", leave=False)
                          
    with torch.no_grad():
        for batch_X, batch_y in val_loader_tqdm:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)

            loss = criterion(logits, batch_y)
            val_loss += loss.item()

            # Apply softmax to get class probabilities in inference
            probs = torch.softmax(logits, dim=1)  # shape: [B, C, H, W]

            # Ground truth class labels should also be [B, H, W]
            targets = batch_y.long()

            val_loader_tqdm.set_postfix(loss=val_loss / (val_loader_tqdm.n + 1))

        avg_train_loss = epoch_loss / len(train_loader)

        avg_val_loss = val_loss / len(val_loader)
                          
        # Step the scheduler based on val loss
        scheduler.step(avg_val_loss)
                          
        # step the scheduler based on epoch
        #scheduler.step(epoch)
        
        # print training updates
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save model to file directory
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config["save_loc"]+config["weights_fname"])
                                       
                          
