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

## function to convert x/y physical coords in GOES grid to lat/lon points
## taken from Da's ResNet work
def xy_to_latlon(x,y):
    req  = 6378137.
    rpol = 6356752.31414
    H    = 42164160.
    lambda0 =  -1.308996939#1000#-1.308996939#-1.56206968053#-1.308996939#-1.56206968053
    a = np.power(np.sin(x),2) + np.power(np.cos(x),2)*(np.power(np.cos(y),2)+(np.power(req,2))/(np.power(rpol,2))*np.power(np.sin(y),2))
    b = -2*H*np.cos(x)*np.cos(y)
    c = np.power(H,2) - np.power(req,2)
#    print( (a,b,c))
    #print(np.power(b,2)-4*a*c)
    rs = (-b-np.sqrt(np.power(b,2)-4*a*c))/(2*a)
    sx =  rs*np.cos(x)*np.cos(y)
    sy = -rs*np.sin(x)
    sz =  rs*np.cos(x)*np.sin(y)
#    print( (rs, sx, sy, sz) )
    lat = ( np.arctan((np.power(req,2))/(np.power(rpol,2))*sz/np.sqrt(np.power((H-sx),2)+np.power(sy,2))) ) * 180. / np.pi
    lon = (lambda0-np.arctan((sy)/(H-sx)) ) * 180. / np.pi
    return (lat, lon)

## adjust lat/lon coords to a new location based on required parallax correction
## taken from Da's ResNet work
def reckon(lat, lon, d, az):
    R = 6378.137 #Radius of the Earth
    brng = np.deg2rad(az) #Bearing is 90 degrees converted to radians.
    #d: Distance in km

    #lat2  52.20444 - the lat result I'm hoping for
    #lon2  0.36056 - the long result I'm hoping for.

    lat1 = np.deg2rad(lat) #Current lat point converted to radians
    lon1 = np.deg2rad(lon) #Current long point converted to radians

    lat2 = np.arcsin( np.sin(lat1)*np.cos(d/R) +
         np.cos(lat1)*np.sin(d/R)*np.cos(brng))

    lon2 = lon1 + np.arctan2(np.sin(brng)*np.sin(d/R)*np.cos(lat1),np.cos(d/R)-np.sin(lat1)*np.sin(lat2))

    lat2 = np.rad2deg(lat2)
    lon2 = np.rad2deg(lon2)

    return lat2, lon2

## perform parallax correction on the gridded GOES-16/19 data
## taken from Da's ResNet work
def parallax_correction(ds_acha, lat, lon, index_ACHA, lat_ACHA, lon_ACHA):
    
    R=6371
    r=35786+6371
    lon0=-75
    
    ## get the data and clean it up
    
    htlat = lat_ACHA[index_ACHA]
    htlon = lon_ACHA[index_ACHA]

    HT = ds_acha.HT.values
    HT = HT[index_ACHA]
    HT[HT<=0]=0 #20000+HT[HT<=0]
    HT[np.isnan(HT)==1]=0
    #HT[HT.mask==True]=0

    ## do the parallax correction
    
    gamma=np.arccos(np.cos(np.deg2rad(htlat))*np.cos(np.deg2rad(lon0-htlon)))
    z=np.arctan(r*np.sin(gamma)/(r*np.cos(gamma)-R))
    d=HT*np.tan(z)/1000
    beta=np.arccos(1/np.tan(gamma)*np.tan(np.deg2rad(htlat)))
    htlatc, htlonc = reckon(htlat,htlon,d,180-beta)
    dhtlat=htlatc-htlat
    dhtlon=htlonc-htlon

    dlat = interpolate.griddata((htlon,htlat),dhtlat,(lon,lat), method='linear')
    dlon = interpolate.griddata((htlon,htlat),dhtlon,(lon,lat), method='linear')
    
    lonc=lon+dlon
    latc=lat+dlat
    
    return latc, lonc

## grab GOES data for the desired inference time (closest available to the inference time)
def get_GOES_single_time(valid_time):

    datetime = valid_time

    # ABI Multi-Channel Cloud Moisture Imagery Product
    # if using goes-19 data need to change satellite argument to "19"
    G = GOES(satellite=16, product="ABI-L2-MCMIPC", domain='C')
    G_acha = GOES(satellite=16, product="ABI-L2-ACHAC", domain='C')
    
    ds = G.nearesttime(datetime)
    ds2 = G_acha.nearesttime(datetime)
    
    BT_channel_7 = ds.CMI_C07.values
    BT_channel_8 = ds.CMI_C08.values
    BT_channel_9 = ds.CMI_C09.values
    BT_channel_10 = ds.CMI_C10.values
    BT_channel_11 = ds.CMI_C11.values
    BT_channel_12 = ds.CMI_C12.values
    BT_channel_13 = ds.CMI_C13.values
    BT_channel_14 = ds.CMI_C14.values
    BT_channel_15 = ds.CMI_C15.values
    BT_channel_16 = ds.CMI_C16.values
    
    predictor = np.stack([BT_channel_7, BT_channel_8, BT_channel_9, BT_channel_10, BT_channel_11, BT_channel_12, BT_channel_13, BT_channel_14, BT_channel_15, BT_channel_16], axis=-1)
    
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]
    xx,yy = np.meshgrid(x,y)
        
    lat_GOES, lon_GOES = xy_to_latlon(xx,yy)
    
    x = ds2.variables['x'][:]
    y = ds2.variables['y'][:]
    xx,yy = np.meshgrid(x,y)
    
    lat_ACHA, lon_ACHA = xy_to_latlon(xx,yy)
    
    index_ACHA = np.ones((300, 500), dtype=bool)
    
    lat_GOES[np.isnan(lat_GOES)==1]=25555
    lon_GOES[np.isnan(lon_GOES)==1]=25555
    
    lat_ACHA[np.isnan(lat_ACHA)==1]=25555
    lon_ACHA[np.isnan(lon_ACHA)==1]=25555
    
    latgrid_corrected, longrid_corrected = parallax_correction(ds2, lat_GOES, lon_GOES, index_ACHA, lat_ACHA, lon_ACHA)

    latgrid_corrected[np.isnan(latgrid_corrected)==1]=25555
    longrid_corrected[np.isnan(longrid_corrected)==1]=25555

    return longrid_corrected, latgrid_corrected, predictor

## grab MRMS data from AWS for the desired inference time (closest available to the inference time)
def grab_MRMS_single_time(valid_time):

    datetime_obj = valid_time

    condensed_dtime = datetime_obj[0:4]+datetime_obj[5:7]+datetime_obj[8:10]
    time = datetime_obj[11:13]+datetime_obj[14:16]+'00'

    fs = s3fs.S3FileSystem(anon=True)
    
    aws_dir = 's3://noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/'+condensed_dtime+'/'
    file_list = fs.ls(aws_dir)

    time_target = time
    time_target = datetime.strptime(time_target, "%H%M%S")
    
    differences = []
    
    for j in range(len(file_list)):
        file_time = file_list[j][-15:-9]
        file_time = datetime.strptime(file_time, "%H%M%S")
        difference = abs((file_time - time_target).total_seconds())
        differences.append(difference)
    
    diff_idx = np.argmin(differences)
    target_file = file_list[diff_idx]
    
    aws_url = 's3://'+target_file

    with fs.open(aws_url, "rb") as fileObj:
        with gzip.GzipFile(fileobj=fileObj, mode='rb') as decompressed:
            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as tmpfile:
                tmpfile.write(decompressed.read()) 
                tmpfile.flush() 
                
                ds_mrms = xr.open_dataset(tmpfile.name, engine="cfgrib")

                refc = ds_mrms.unknown.values
                lon = ds_mrms.longitude.values
                lat = ds_mrms.latitude.values

    return lon, lat, refc

## build an interpolator for the GOES data surrounding the desired prediction patch
## goal is to use this interpolator to eventually place the GOES data (2km, curvilinear) onto the MRMS grid (~1km lat/lon)
def create_GOES_interpolator(lon_GOES, lat_GOES, patch_lon, patch_lat):
    
    # Precompute Delaunay triangulation and interpolator using the GOES grid.
    
    # Flatten all 2D arrays
    lon_flat = lon_GOES.ravel()
    lat_flat = lat_GOES.ravel()

    # Define bounding box with buffer
    lon_min = patch_lon.min() - 1
    lon_max = patch_lon.max() + 1
    lat_min = patch_lat.min() - 1
    lat_max = patch_lat.max() + 1

    # Apply bounding box mask
    mask = (
        (lon_flat >= lon_min) & (lon_flat <= lon_max) &
        (lat_flat >= lat_min) & (lat_flat <= lat_max)
    )

    if np.count_nonzero(mask) < 10:
        raise ValueError("Too few points left after cropping for triangulation.")

    # Crop points
    lon_cropped = lon_flat[mask]
    lat_cropped = lat_flat[mask]

    goes_points = np.column_stack((lon_cropped, lat_cropped))
    triangulation = Delaunay(goes_points)

    def build_interpolator(values_2d, mask):
        values_1d = values_2d.ravel()
        values_1d = values_1d[mask]
        return LinearNDInterpolator(triangulation, values_1d, fill_value=np.nan)

    return build_interpolator, mask

## build a target area on the MRMS grid to interpolate the GOES data onto
## target region is within the desired patch prediction area
def create_MRMS_target_grid(lon_mrms, lat_mrms, patch_lon, patch_lat):
    
    # Create cropped MRMS meshgrid for interpolation targets.
    
    lon_min = patch_lon.min() - 0.5
    lon_max = patch_lon.max() + 0.5
    lat_min = patch_lat.min() - 0.5
    lat_max = patch_lat.max() + 0.5
    
    lon_mask = (lon_mrms >= (lon_min + 360)) & (lon_mrms <= (lon_max + 360))
    lat_mask = (lat_mrms >= lat_min) & (lat_mrms <= lat_max)

    lon_cropped = lon_mrms[lon_mask]
    lat_cropped = lat_mrms[lat_mask]

    lon_grid, lat_grid = np.meshgrid(lon_cropped - 360, lat_cropped)
    target_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    
    return lon_grid.shape, target_points, lon_grid, lat_grid

## use the cached interpolator from create_GOES_interpolator() to interpolate each GOES grid onto the MRMS target grid/points
def regrid_with_cached_interpolator(interpolator_func, goes_values, target_shape, target_points, mask):
    
    # Interpolate one GOES timestep using pre-built triangulation.
    
    interpolator = interpolator_func(goes_values, mask)
    regridded = interpolator(target_points).reshape(target_shape)
    return regridded

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
    
def inference(start_date, periods, lat0, lon0, patch_width, weights_path, config_path, scaler_path, leadtime, class_to_plot, save_path):

    CI_times = pd.date_range(start=start_date, periods=periods, freq="5min")
    
    interval = 0.01 ## ~1km grid spacing
    patch_lat, patch_lon = np.meshgrid(np.arange(lat0-patch_width/2,lat0+patch_width/2-1e-10,interval),np.arange(lon0-patch_width/2,lon0+patch_width/2-1e-10,interval)) ## meshgrid construction aligning with Da's ResNet patch construction

    ## loop through datetimes, grab data for each, run the model, and then plot
    for i in range(len(CI_times)):
    
        CI_time = CI_times[i]
        
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

        if class_to_plot == 'CI':
            class_idx = 0
        if class_to_plot == 'MC':
            class_idx = 1
        if class_to_plot == 'HC':
            class_idx = 2
        if class_to_plot == 'EC':
            class_idx = 3
            
        probs_CI = probs[0,class_idx,:,:] # for CI: change second argument if you want other classes... (0 = CI, 1 = Minimal Clouds, 2 = High Clouds, 3 = EC)

        ## can include counties if you wish --> helps with the local zoom to provide some context!
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
        
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(131, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(132, projection=ccrs.PlateCarree())
        ax3 = fig.add_subplot(133, projection=ccrs.PlateCarree())
        
        ax.set_extent([patch_lon.min(),patch_lon.max(),patch_lat.min(),patch_lat.max()], crs=ccrs.PlateCarree())
        ax2.set_extent([patch_lon.min(),patch_lon.max(),patch_lat.min(),patch_lat.max()], crs=ccrs.PlateCarree())
        ax3.set_extent([patch_lon.min(),patch_lon.max(),patch_lat.min(),patch_lat.max()], crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        ax.add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        #ax.add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.2, zorder=5)
        
        ax2.add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        ax2.add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        ax2.add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        #ax2.add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.2, zorder=5)
        
        ax3.add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        ax3.add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        ax3.add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        #ax3.add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.2, zorder=5)
    
        mrms_grid = median_filter(refc_data, size=3)
            
        contf = ax.contourf(patch_lon, patch_lat, gaussian_filter(probs_CI*100, sigma=2.0), levels=np.arange(0,105,5), cmap=prob_cmap, zorder=1, transform=ccrs.PlateCarree())
        contf_GOES = ax2.pcolormesh(patch_lon, patch_lat, median_filter(BTs_all[6,:,:], size=5), vmin=200, vmax=350, cmap=cmap, zorder=1, transform=ccrs.PlateCarree())
        contf_MRMS = ax3.pcolormesh(lon, lat, np.ma.masked_array(mrms_grid, mask=(mrms_grid==0)), cmap=rain_cmap, norm=rain_norm, zorder=1, transform=ccrs.PlateCarree())
    
        contour_levels = [20,40,60,80]
        contour_colors = ['#f0b3f4', '#dc74e3', '#c63dcf', '#a017a9']
    
        ax2.contour(patch_lon, patch_lat, gaussian_filter(probs_CI*100, sigma=2.0), levels=contour_levels, colors=contour_colors, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
        ax3.contour(patch_lon, patch_lat, gaussian_filter(probs_CI*100, sigma=2.0), levels=contour_levels, colors=contour_colors, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
    
        cax = inset_axes(ax, width='100%', height='3.25%', loc='lower left',
                    bbox_to_anchor=(0,-0.045,1,1), bbox_transform=ax.transAxes, borderpad=0)
        
        cbar = plt.colorbar(contf, orientation='horizontal', extendfrac=0, shrink=0.19, cax=cax)
        cbar.set_ticks(np.arange(0,110,10))
        cax.tick_params(labelsize=12) 
    
        cax = inset_axes(ax2, width='100%', height='3.25%', loc='lower left',
                    bbox_to_anchor=(0,-0.045,1,1), bbox_transform=ax2.transAxes, borderpad=0)
        
        cbar = plt.colorbar(contf_GOES, orientation='horizontal', extendfrac=0, shrink=0.19, cax=cax)
        cbar.set_ticks(np.arange(200,375,25))
        cax.tick_params(labelsize=12) 
    
        cax = inset_axes(ax3, width='100%', height='3.25%', loc='lower left',
                    bbox_to_anchor=(0,-0.045,1,1), bbox_transform=ax3.transAxes, borderpad=0)
        
        cbar = plt.colorbar(contf_MRMS, orientation='horizontal', extendfrac=0, shrink=0.19, cax=cax)
        cbar.set_ticks(rain_levels)
        cax.tick_params(labelsize=12) 
    
        ## can adjust title information if you wish, and depending on leadtime/class
        ax.set_title('U-Net '+leadtime[0:2]+'-'+leadtime[2:5]+' '+class_to_plot+' Probability (%)\n'+r"$\bf{Init}$"+": "+str(CI_time)[11:16]+'z '+pd.to_datetime(CI_time).strftime('%b %-d, %Y'), loc='center', fontsize=14, pad=8.5)    
        ax2.set_title('Channel 13 BT (K) & CI Probability (purple)\n'+r"$\bf{Valid}$"+": "+str(CI_time)[11:16]+'z '+pd.to_datetime(CI_time).strftime('%b %-d, %Y'), loc='center', fontsize=14, pad=8.5)
        ax3.set_title('REFC (dBZ) & CI Probability (purple)\n'+r"$\bf{Valid}$"+": "+str(CI_time)[11:16]+'z '+pd.to_datetime(CI_time).strftime('%b %-d, %Y'), loc='center', fontsize=14, pad=8.5)
    
        plt.tight_layout(w_pad=2.5)
    
        ## savefig with specified title and index corresponding to interval from CI_times
        ## would be good to adjust this to include the datetime automatically!
        plt.savefig(save_path+'unet_'+class_to_plot+'_'+leadtime+'_frame'+str(i)+'.png', bbox_inches='tight', pad_inches=0.25, facecolor='white')


####### EDIT BELOW THIS LINE

leadtime = '10min' ## adjust leadtime (10min, 20min, 30min, or 40min)
config_path = "../config/" ## path to configs folder (10min, 20min, 30min, or 40min), fnames should match the ones in Github repo
weights_path = "../weights/" ## path to weights folder, fnames found in config
scaler_path = "../weights/" ## path to scalers folder, fnames found in config
start_date = "2024-05-06 18:30" ## start predictions at this time (inclusive)
periods_to_run = 5 ## 5min intervals to run out from the start date
class_to_plot = 'CI' ## options: 'CI', 'EC' (existing convection), 'HC' (high cloud), 'MC' (minimal cloud)
save_path = '../figures/' ## save plots here

lat0 = 37.9 ## central lat for patch
lon0 = -99.93 ## central lon for patch

patch_width = 2.56 ## we will run with a patch size of 256x256km here...this can be changed to any multiple of 128x128km, although it's likely more stable to run without a massive size...
    
inference(start_date, periods_to_run, lat0, lon0, patch_width, weights_path, config_path, scaler_path, leadtime, class_to_plot, save_path)
