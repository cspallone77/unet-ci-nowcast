import torch
from lightning.pytorch import seed_everything
import xarray as xr
import numpy as np
import joblib
import yaml
from model import CI_unet

seed_everything(77)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(config_fname) as config_file:
    config = yaml.safe_load("/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/training_data/training_config/CI_unet_10min.yml")
            
model = CI_unet(**config["model"])

checkpoint = torch.load("/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/model_weights/unet_10min.pth", map_location="cpu")

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

scaler = joblib.load("/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/model_weights/scalers/scaler_UNet_10min.bin") 

test_data = np.load('/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/10min/test_predictors_10min.npy')

test_label_ds = xr.open_dataset('/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/all_test_labels.nc')

test_data = np.transpose(test_data, (0, 2, 3, 1))  
test_data_scaled = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape) 
test_data_scaled = np.transpose(test_data_scaled, (0, 3, 1, 2))

## these calculated nans are also found in: "/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/10min/nan_indices_test_10min.npy"
nan_indices = np.argwhere(np.isnan(test_data))
sample_indices_with_nan = np.unique(nan_indices[:, 0])

test_label = np.delete(test_label, sample_indices_with_nan, axis=0)
test_data_scaled = np.delete(test_data_scaled, sample_indices_with_nan, axis=0)

test_data_scaled = torch.tensor(test_data_scaled, dtype=torch.float32)
test_label = torch.tensor(test_label, dtype=torch.long)

# X.shape: (B, C, H, W), y.shape: (B, H, W)
X = test_data_scaled.to('cpu')  # predictors
y = test_label.to('cpu')  # ground truth

# Run prediction
with torch.no_grad():
    logits = model(X)  # shape: (B, num_classes, H, W)
    probs = torch.softmax(logits, dim=1)
    #probs = torch.sigmoid(logits)

probs_np = probs.cpu().numpy()

## save output to your desired directory
#np.save('/storage/group/sjg213/default/cps5892/U-Net/testing_files_full/nan_indices.npy', np.array(sample_indices_with_nan))
#np.save('/storage/group/sjg213/default/cps5892/U-Net/testing_files_full/probs.npy', np.array(probs_np))