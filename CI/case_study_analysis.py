from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.io.shapereader as shpreader
from metpy.plots import ctables
import matplotlib
import matplotlib.pyplot as plt
import s3fs
import tempfile
import gzip
from datetime import datetime, timedelta
from scipy.interpolate import griddata, RegularGridInterpolator, LinearNDInterpolator
from scipy import interpolate 
from goes2go import GOES
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter
import joblib
import torch
from scipy.spatial import Delaunay
import argparse
import yaml
import sys
from model import CI_unet
from satrad_preprocess import get_GOES_single_time, grab_MRMS_single_time, create_GOES_interpolator, create_MRMS_target_grid, regrid_with_cached_interpolator
from herbie import Herbie

## function to make a single timestep prediction (using all GOES predictor data) with the CI UNET model
def make_unet_pred(predictors, scaler_fname, weights_fname, config):

    predictors = predictors[np.newaxis,:,:,:]
    predictors = np.transpose(predictors, (0, 2, 3, 1))

    scaler = joblib.load(scaler_fname)
    predictors_scaled = scaler.transform(predictors.reshape(-1, predictors.shape[-1])).reshape(predictors.shape)

    predictors_scaled = np.transpose(predictors_scaled, (0, 3, 1, 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## these arguments will be grabbed and inputted based on the config file
    model = CI_unet(**config["model"])

    ## load weights
    ## may have to adjust directory depending on where your weights are
    model.load_state_dict(torch.load(weights_fname, map_location=torch.device('cpu'))) 

    model.eval()
    model.to('cpu')

    predictors_scaled = torch.tensor(predictors_scaled, dtype=torch.float32)
    
    X = predictors_scaled.to('cpu')  

    # Run prediction
    with torch.no_grad():
        logits = model(X)  # shape: (B, num_classes, H, W)
        probs = torch.softmax(logits, dim=1)
        #probs = torch.sigmoid(logits) ## for binary model

    CI_probs = probs.cpu().numpy()

    return CI_probs

def run_unet(time_to_run, lat0, lon0, patch_width, weights_path, config_path, scaler_path, leadtime):
    
    interval = 0.01 ## ~1km grid spacing
    patch_lat, patch_lon = np.meshgrid(np.arange(lat0-patch_width/2,lat0+patch_width/2-1e-10,interval),np.arange(lon0-patch_width/2,lon0+patch_width/2-1e-10,interval)) ## meshgrid construction aligning with Da's ResNet patch construction
    
    CI_time = time_to_run
    
    lon, lat, refc_data = grab_MRMS_single_time(str(CI_time))
    lon_full, lat_full, GOES_data = get_GOES_single_time(str(CI_time))

    build_interpolator, data_mask = create_GOES_interpolator(lon_full, lat_full, patch_lon, patch_lat)
    target_shape, target_points, longitudes_predictors, latitudes_predictors = create_MRMS_target_grid(lon, lat, patch_lon, patch_lat)
            
    for j in range(GOES_data.shape[-1]):
    
        GOES_data_sample = GOES_data[:,:,j]
        
        predictor_regridded = regrid_with_cached_interpolator(build_interpolator, GOES_data_sample, target_shape, target_points, data_mask)
        
        BTs = predictor_regridded
        lat_1d = latitudes_predictors[:, 0] 
        lon_1d = longitudes_predictors[0, :] 

        interp_fn_goes = RegularGridInterpolator((lat_1d, lon_1d), BTs, method='linear', bounds_error=False, fill_value=np.nan)
        
        # Interpolate at patch points (must be 2D arrays)
        points = np.column_stack([patch_lat.ravel(), patch_lon.ravel()])
        BT_cropped = interp_fn_goes(points).reshape(patch_lat.shape)
        
        if j == 0:
            BTs_all = BT_cropped
            BTs_all = BTs_all.reshape(1, 256, 256)
        else:
            BT_cropped = BT_cropped.reshape(1, 256, 256)
            BTs_all = np.concatenate((BTs_all, BT_cropped), axis=0)

    config_fname = config_path+'CI_unet_'+leadtime+'.yml'

    ## load config (from config directory)
    with open(config_fname) as config_file:
        config = yaml.safe_load(config_file)

    scaler_fname = scaler_path+config["scaler_fname"]
    weights_fname = weights_path+config["weights_fname"]

    probs = make_unet_pred(BTs_all, scaler_fname, weights_fname, config)

    return probs, BTs_all, refc_data, lon, lat
        
## makes a 6x6 panel case study layout to analyze U-Net predictions
def make_case_study_plot(start_date, lat0, lon0, patch_width, weights_path, config_path, scaler_path, save_path):
        
    fig, axes = plt.subplots(6, 6, figsize=(20, 15), sharex=True, sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
    
    interval = 0.01
    patch_lat, patch_lon = np.meshgrid(np.arange(lat0-patch_width/2,lat0+patch_width/2-1e-10,interval),np.arange(lon0-patch_width/2,lon0+patch_width/2-1e-10,interval))
    
    CI_times = pd.date_range(start=start_date, periods=6, freq="15min") ## fixed 6 15-min intervals for this plot
    
    start = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
    titles = [(start + timedelta(minutes=15*i)).strftime("%H:%M") for i in range(6)]
    
    #reader = shpreader.Reader('../Shapefiles/ne_10m_admin_2_counties.shp')
    #counties = list(reader.geometries())
    #COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    
    norm, cmap = ctables.registry.get_with_range("WVCIMSS_r", 200, 350)
    
    probability_colors = ['#FFFFFF','#E7EAED','#D4DAE1','#C2C9D2','#A3BDDF','#94B9E8','#7DADE9','#68A1E9','#7EDD9E','#67D48C','#40D371','#27C85C','#FAF248','#FADA48','#FAB748','#FAAC48','#FA9648','#FA7648','#FA5548','#D73529','#A6180E']
    prob_cmap = matplotlib.colors.ListedColormap(probability_colors)
    
    rain_colors = ['#FFFFFF','#00F747','#02DB41','#01B636','#038B2A','#F5FF1E','#FFD31E','#FFAA1E','#FF691E','#FF1E1E','#BE2020','#FF28DB']
    rain_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]
    rain_cmap = matplotlib.colors.ListedColormap(rain_colors)
    rain_norm = matplotlib.colors.BoundaryNorm(rain_levels,len(rain_levels))

    nearest_hour_lower = start.replace(minute=0, second=0)

    H = Herbie(nearest_hour_lower, model='hrrr', product="subh", fxx=1)
    H2 = Herbie(nearest_hour_lower, model='hrrr', product="subh", fxx=2)
    H3 = Herbie(nearest_hour_lower, model='hrrr', product="subh", fxx=3)
    
    ds = H.xarray(":REFC:")
    ds2 = H2.xarray(":REFC:")
    ds3 = H3.xarray(":REFC:")
    
    refc = ds.refc.values
    times = ds.valid_time.values
    refc = refc.reshape((4, refc.shape[1], refc.shape[2]))
    
    refc2 = ds2.refc.values
    times2 = ds2.valid_time.values
    refc2 = refc2.reshape((4, refc2.shape[1], refc2.shape[2]))
    
    refc3 = ds3.refc.values
    times3 = ds3.valid_time.values
    refc3 = refc3.reshape((4, refc3.shape[1], refc3.shape[2]))

    all_refc = np.concatenate((refc, refc2, refc3), axis=0)
    all_times = np.concatenate((times, times2, times3), axis=0)

    all_times = pd.to_datetime(all_times).strftime("%H:%M")
    matched_idxs = [np.where(all_times == t)[0][0] for t in titles]

    all_refc = all_refc[matched_idxs]
    
    for i in range(len(CI_times)):

        CI_time = CI_times[i]
    
        probs_10min, BTs_all, refc_data, mrms_lon, mrms_lat = run_unet(CI_time, lat0, lon0, patch_width, weights_path, config_path, scaler_path, '10min')
        probs_20min, BTs_all, refc_data, mrms_lon, mrms_lat = run_unet(CI_time, lat0, lon0, patch_width, weights_path, config_path, scaler_path, '20min')
        
        probs_CI_10min = probs_10min[0,0,:,:] 
        probs_CI_20min = probs_20min[0,0,:,:] 
        probs_EC_20min = probs_20min[0,3,:,:] 
        
        axes[0,i].set_extent([patch_lon.min(),patch_lon.max(),patch_lat.min(),patch_lat.max()], crs=ccrs.PlateCarree())
       
        axes[0,i].add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[0,i].add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[0,i].add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        #axes[0,i].add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.2, zorder=5)
    
        axes[1,i].set_extent([patch_lon.min(),patch_lon.max(),patch_lat.min(),patch_lat.max()], crs=ccrs.PlateCarree())
       
        axes[1,i].add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[1,i].add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[1,i].add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        #axes[1,i].add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.2, zorder=5)
    
        axes[2,i].set_extent([patch_lon.min(),patch_lon.max(),patch_lat.min(),patch_lat.max()], crs=ccrs.PlateCarree())
       
        axes[2,i].add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[2,i].add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[2,i].add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        #axes[2,i].add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.2, zorder=5)
    
        axes[3,i].set_extent([patch_lon.min(),patch_lon.max(),patch_lat.min(),patch_lat.max()], crs=ccrs.PlateCarree())
       
        axes[3,i].add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[3,i].add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[3,i].add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        #axes[3,i].add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.2, zorder=5)
    
        axes[4,i].set_extent([patch_lon.min(),patch_lon.max(),patch_lat.min(),patch_lat.max()], crs=ccrs.PlateCarree())
       
        axes[4,i].add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[4,i].add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[4,i].add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        #axes[4,i].add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.2, zorder=5)
    
        axes[5,i].set_extent([patch_lon.min(),patch_lon.max(),patch_lat.min(),patch_lat.max()], crs=ccrs.PlateCarree())
       
        axes[5,i].add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[5,i].add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        axes[5,i].add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        #axes[5,i].add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.2, zorder=5)
    
        mrms_grid = median_filter(refc_data, size=3) 
        
        contf_GOES = axes[5,i].pcolormesh(patch_lon, patch_lat, median_filter(BTs_all[6,:,:], size=5), vmin=200, vmax=350, cmap=cmap, zorder=1, transform=ccrs.PlateCarree())
        contf_MRMS = axes[4,i].pcolormesh(mrms_lon, mrms_lat, np.ma.masked_array(mrms_grid, mask=(mrms_grid==0)), cmap=rain_cmap, norm=rain_norm, zorder=1, transform=ccrs.PlateCarree())
    
        contf_HRRR = axes[3,i].pcolormesh(ds.longitude.values, ds.latitude.values, np.ma.masked_array(all_refc[i,:,:], mask=(all_refc[i,:,:]==0)), cmap=rain_cmap, norm=rain_norm, zorder=1, transform=ccrs.PlateCarree())
    
        contour_levels = [20,40,60,80]
        contour_colors = ['#f0b3f4', '#dc74e3', '#c63dcf', '#a017a9']
    
        axes[5,i].contour(patch_lon, patch_lat, gaussian_filter(probs_CI_10min*100, sigma=2.0), levels=contour_levels, colors=contour_colors, linewidths=1, zorder=1, transform=ccrs.PlateCarree()) ## was LW=1
        axes[4,i].contour(patch_lon, patch_lat, gaussian_filter(probs_CI_10min*100, sigma=2.0), levels=contour_levels, colors=contour_colors, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
    
        contf_CI_10min = axes[0,i].contourf(patch_lon, patch_lat, gaussian_filter(probs_CI_10min*100, sigma=2.0), levels=np.arange(0,105,5), cmap=prob_cmap, zorder=1, transform=ccrs.PlateCarree())
        contf_EC_20min = axes[2,i].contourf(patch_lon, patch_lat, gaussian_filter(probs_EC_20min*100, sigma=2.0), levels=np.arange(0,105,5), cmap=prob_cmap, zorder=1, transform=ccrs.PlateCarree())
        contf_CI_20min = axes[1,i].contourf(patch_lon, patch_lat, gaussian_filter(probs_CI_20min*100, sigma=2.0), levels=np.arange(0,105,5), cmap=prob_cmap, zorder=1, transform=ccrs.PlateCarree())
    
        axes[0,i].set_title(titles[i], fontsize=22, loc='center', pad=12.5)
    
        if i == 5:
            cax = inset_axes(axes[0,i], width='3%', height='100%', loc='lower right', bbox_to_anchor=(0.045,0,1,1), bbox_transform=axes[0,i].transAxes, borderpad=0)
            cbar = plt.colorbar(contf_CI_10min, orientation='vertical', extendfrac=0, cax=cax)
            cbar.set_ticks(np.arange(10,110,20))
            cbar.set_label("Probability (%)", fontsize=13, labelpad=20, rotation=270)
            cax.tick_params(labelsize=13)
        
            cax = inset_axes(axes[1,i], width='3%', height='100%', loc='lower right', bbox_to_anchor=(0.045,0,1,1), bbox_transform=axes[1,i].transAxes, borderpad=0)
            cbar = plt.colorbar(contf_CI_20min, orientation='vertical', extendfrac=0, cax=cax)
            cbar.set_ticks(np.arange(10,110,20))
            cbar.set_label("Probability (%)", fontsize=13, labelpad=20, rotation=270)
            cax.tick_params(labelsize=13)
        
            cax = inset_axes(axes[2,i], width='3%', height='100%', loc='lower right', bbox_to_anchor=(0.045,0,1,1), bbox_transform=axes[2,i].transAxes, borderpad=0)
            cbar = plt.colorbar(contf_EC_20min, orientation='vertical', extendfrac=0, cax=cax)
            cbar.set_ticks(np.arange(10,110,20))
            cbar.set_label("Probability (%)", fontsize=13, labelpad=20, rotation=270)
            cax.tick_params(labelsize=13)
        
            cax = inset_axes(axes[4,i], width='3%', height='100%', loc='lower right', bbox_to_anchor=(0.045,0,1,1), bbox_transform=axes[4,i].transAxes, borderpad=0)
            cbar = plt.colorbar(contf_MRMS, orientation='vertical', extendfrac=0, cax=cax)
            cbar.set_label("Reflectivity (dBZ)", fontsize=13, labelpad=20, rotation=270)
            cax.tick_params(labelsize=13)
        
            cax = inset_axes(axes[5,i], width='3%', height='100%', loc='lower right', bbox_to_anchor=(0.045,0,1,1), bbox_transform=axes[5,i].transAxes, borderpad=0)
            cbar = plt.colorbar(contf_GOES, orientation='vertical', extendfrac=0, cax=cax)
            cbar.set_ticks(np.arange(200,360,20)) 
            cbar.set_label("BT (K)", fontsize=13, labelpad=20, rotation=270)
            cax.tick_params(labelsize=13)
    
            cax = inset_axes(axes[3,i], width='3%', height='100%', loc='lower right', bbox_to_anchor=(0.045,0,1,1), bbox_transform=axes[3,i].transAxes, borderpad=0)
            cbar = plt.colorbar(contf_HRRR, orientation='vertical', extendfrac=0, cax=cax)
            cbar.set_label("Reflectivity (dBZ)", fontsize=13, labelpad=20, rotation=270)
            cax.tick_params(labelsize=13)
        
    wavelengths = ["10-min (CI)", "20-min (CI)", "20-min (EC)", titles[0][0:2]+"z HRRR REFC", "MRMS REFC", "CH13 BT"]
    
    for i, wavelength in enumerate(wavelengths):
        ax = axes[i, 0]  # Leftmost column in each row
        ax.text(0.05, 0.13, wavelength, transform=ax.transAxes,
                fontsize=12.5, zorder=15, fontweight='bold', verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.50'))
    
    plt.tight_layout(w_pad=-32, h_pad=1)
    
    plt.savefig(save_path+'case_study_layout.png', bbox_inches='tight', pad_inches=0.25, facecolor='w')

####### EDIT BELOW THIS LINE

config_path = "../config/" ## path to configs folder (10min, 20min, 30min, or 40min), fnames should match the ones in Github repo
weights_path = "../weights/" ## path to weights folder, fnames found in config
scaler_path = "../weights/" ## path to scalers folder, fnames found in config
save_path = '../figures/' ## save plots here

start_date = "2024-05-06 18:15" ## start predictions at this time (inclusive)

lat0 = 37.90 ## central lat for patch
lon0 = -99.93 ## central lon for patch

patch_width = 2.56 ## we will run with a patch size of 256x256km here...this can be changed to any multiple of 128x128km, although it's likely more stable to run without a massive size...
    
make_case_study_plot(start_date, lat0, lon0, patch_width, weights_path, config_path, scaler_path, save_path)
