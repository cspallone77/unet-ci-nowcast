import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import generate_binary_structure, label
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

## function to make a histogram of CI object areas in an xarray dataset
## finds the area of objects from the multiclass labels in all gridded samples and plots them
def make_area_histogram(ds_labels_path, save_path):

    ## open labels dataset
    ds = xr.open_dataset(ds_labels_path)

    all_labels = ds.labels.values
    all_times = ds.time.values
    
    all_sizes_events = []

    ## make list with all object sizes
    for i in range(len(all_times)):
        
        CI_event_mask = (all_labels[i,:,:] == 0)
        s = generate_binary_structure(2,2)
        labeled_CI_events, num_CI_events = label(CI_event_mask, structure=s)
    
        if num_CI_events > 0:
            for region_label in range(1, num_CI_events + 1):
                region_mask = (labeled_CI_events == region_label)
                region_size = np.sum(region_mask)
                all_sizes_events.append(region_size)

    ## make figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.xlabel("Object Area (km$^2$)", fontsize=20, labelpad=20)
    plt.ylabel("CI Object Count", fontsize=20, labelpad=20)
               
    sns.histplot(all_sizes_events, stat='count', kde=False, binwidth=2, binrange=(10,70), zorder=1, ax=ax, color='mediumseagreen')
    
    ax.grid(linestyle='--', axis='y', color='gray', linewidth=0.6, zorder=5)
    
    ax.set_title("Area Histogram of CI Objects in the Training Dataset", loc='center', fontsize=22, pad=15)

    ## save figure
    plt.savefig(save_path+"CI_object_area.png", bbox_inches='tight', pad_inches=0.25, facecolor='w')

## function to make a histogram of CI event occurence hour 
## finds the hour of occurence of CI objects in the multiclass labels and plots them
def make_hourly_histogram(ds_labels_path, save_path):

    ## open labels dataset
    ds_CI_labels = xr.open_dataset(ds_labels_path)
    all_samples = ds_CI_labels.labels.values

    ## find samples that have CI objects (value=0) in them and remove all others
    ## extract just the times of the CI samples
    mask = (all_samples == 0).any(axis=(1,2)) 
    times = ds_CI_labels.time.values
    times = times[mask]

    ## get times in UTC as a integer representing the hour of day of occurence
    hours = times.astype('datetime64[h]').astype(int) % 24

    ## make figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.xlabel("Hour (UTC)", fontsize=20, labelpad=20)
    plt.ylabel("Sample Count", fontsize=20, labelpad=20)
               
    plt.hist(hours, bins=np.arange(25)-0.5, edgecolor='black', color='teal')  # bins -0.5 to 23.5 to center bars
    plt.xticks(np.arange(0, 24, 2))  
    
    ax.grid(linestyle='--', axis='y', color='gray', linewidth=0.6, zorder=5)
    
    ax.set_title("Histogram of CI Occurence Hour in the Training Dataset", loc='center', fontsize=22, pad=15)
    
    ## save figure
    plt.savefig(save_path+"CI_object_hour.png", bbox_inches='tight', pad_inches=0.25, facecolor='w')

# function to make a density plot of CI event locations by 0.5° bin
# bins the multiclass samples based on their lat/lon patch centroids
def make_density_plot(ds_labels_path, save_path):
    
    ds_CI_labels = xr.open_dataset(ds_labels_path)
    all_samples = ds_CI_labels.labels.values
    lats = ds_CI_labels.lat_0.values
    lons = ds_CI_labels.lon_0.values
    #times = ds_CI_labels.time.values
    
    #time_index = pd.to_datetime(times)
    
    #indices_apr = np.where((time_index.month == 4))[0]
    
    #all_samples = all_samples[indices_apr]
    #lons = lons[indices_apr]
    #lats = lats[indices_apr]
    #times = times[indices_apr]

    ## just get the CI objects
    mask = (all_samples == 0).any(axis=(1,2)) 
    all_samples = all_samples[mask]
    filtered_lats = lats[mask]
    filtered_lons = lons[mask]
    
    # Pair lat/lon into 2D histogram over all possible 0.5 deg bin locations
    lon_bins = np.arange(-104.25, -90.5, 0.5)
    lat_bins = np.arange(30.75, 49.5, 0.5)
    
    # Histogram of event counts per bin
    density, lon_edges, lat_edges = np.histogram2d(filtered_lons, filtered_lats, bins=[lon_bins, lat_bins])
    
    # Transpose for correct orientation
    density_new = density.T
    
    # Plot the density figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([-104.25, -90.75, 30.75, 49.25], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
    ax.add_feature(cfeature.STATES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=1, zorder=10)
    
    # Create mesh of lon/lat bin centeres for pcolormesh
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers) 
    
    # Plot density using pcolormesh
    contf = ax.pcolormesh(lon_mesh, lat_mesh, density_new, vmin=0, vmax=75, cmap='PuRd', shading='auto')
    
    # Add colorbar
    cax = inset_axes(ax, width='4%', height='100%', loc='lower right',
                     bbox_to_anchor=(.07, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(contf, cax=cax)
    cbar.set_ticks(ticks=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75], labels=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75], fontsize=14)
    #cbar.set_ticks(ticks=[0,5,10,15,20,25,30,35,40], labels=[0,5,10,15,20,25,30,35,40], fontsize=14)
    
    #ax.set_title("April", fontsize=40, pad=18)
    ax.set_title("Density of CI Events (0.5° Patches)", fontsize=20, pad=12.5)
    
    plt.savefig(save_path+"CI_density_plot.png", bbox_inches='tight', pad_inches=0.25, facecolor='w')

#ds_labels_path = 'all_train_labels_2022_2023.nc'
#save_path = '../figures/'

#make_area_histogram(ds_labels_path, save_path)
#make_hourly_histogram(ds_labels_path, save_path)
#make_density_plot(ds_labels_path, save_path)
