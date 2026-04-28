from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.io.shapereader as shpreader
from metpy.plots import ctables
import matplotlib
from scipy.spatial import cKDTree
import s3fs
import tempfile
import gzip
from datetime import datetime, timedelta
from goes2go import GOES
import numpy as np
import xarray as xr
import math
from scipy import interpolate 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from geopy import distance
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter, minimum_filter, label, binary_dilation, generate_binary_structure, center_of_mass

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

def get_GOES_timeseries(valid_time):

    all_times = []
    minutes_all_times = [0,5,10,15,20,25,30]
    
    for i in range(7):
        time_to_add = str(pd.to_datetime(valid_time) - timedelta(minutes=minutes_all_times[i]))
        all_times.append(time_to_add)

    for i in range(len(all_times)):

        datetime = all_times[i]

        # ABI Multi-Channel Cloud Moisture Imagery Product
        G = GOES(satellite=16, product="ABI-L2-MCMIPC", domain='C')
        G_acha = GOES(satellite=16, product="ABI-L2-ACHAC", domain='C')
        
        ds = G.nearesttime(datetime)
        ds2 = G_acha.nearesttime(datetime)
        
        BT_channel_13 = ds.CMI_C13.values
        
        predictor = BT_channel_13
        
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

        if i == 0:
            predictor_full = predictor
            predictor_full = predictor_full.reshape(1, 1500, 2500)

            lat_full = latgrid_corrected
            lat_full = lat_full.reshape(1, 1500, 2500)

            lon_full = longrid_corrected
            lon_full = lon_full.reshape(1, 1500, 2500)
            
        else:
            predictor = predictor.reshape(1, 1500, 2500)
            predictor_full = np.vstack((predictor_full, predictor))

            latgrid_corrected = latgrid_corrected.reshape(1, 1500, 2500)
            lat_full = np.vstack((lat_full, latgrid_corrected))

            longrid_corrected = longrid_corrected.reshape(1, 1500, 2500)
            lon_full = np.vstack((lon_full, longrid_corrected))

    return lon_full, lat_full, predictor_full

def get_MRMS_GOES_regrid(lon_mrms, lat_mrms, lon_GOES, lat_GOES, refc, predictor, lon0, lat0, patch_width):

    lon_min = lon0 - (patch_width/2)
    lon_max = lon0 + (patch_width/2)
    lat_min = lat0 - (patch_width/2)
    lat_max = lat0 + (patch_width/2)

    # Crop MRMS (1D lat/lon arrays)
    lon_mask_mrms = (lon_mrms >= (lon_min + 360)) & (lon_mrms <= (lon_max + 360))
    lat_mask_mrms = (lat_mrms >= lat_min) & (lat_mrms <= lat_max)

    lon_cropped_mrms = lon_mrms[lon_mask_mrms]
    lat_cropped_mrms = lat_mrms[lat_mask_mrms]

    refc_cropped = refc[np.ix_(lat_mask_mrms, lon_mask_mrms)]

    # Create MRMS target grid (convert lons back to [-180, 180] range)
    mrms_lon_grid, mrms_lat_grid = np.meshgrid(lon_cropped_mrms - 360, lat_cropped_mrms)

    # Interpolate GOES predictor (which is on a 2D curvilinear grid) to MRMS grid
    points_GOES = np.column_stack((lon_GOES.ravel(), lat_GOES.ravel()))
    values_GOES = predictor.ravel()

    predictor_regridded = griddata(
        points_GOES, values_GOES,
        (mrms_lon_grid, mrms_lat_grid),
        method='linear'
    )

    return refc_cropped, predictor_regridded, mrms_lon_grid, mrms_lat_grid
    
def grab_MRMS_timeseries(valid_time):

    all_times = []
    minutes_all_times = [0,5,10,15,20,25,30]
    
    for i in range(7):
        time_to_add = str(pd.to_datetime(valid_time) - timedelta(minutes=minutes_all_times[i]))
        all_times.append(time_to_add)

    for i in range(len(all_times)):

        datetime_obj = all_times[i]
    
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

        if i == 0:
            refc_full = refc
            refc_full = refc_full.reshape(1, 3500, 7000)
        else:
            refc = refc.reshape(1, 3500, 7000)
            refc_full = np.vstack((refc_full, refc))

    return lon, lat, refc_full

## smooth large convective objects (EC w/ size 5 and CI w/ size 3 median filters)
def image_processing(classes):
    
    cleaned_map = np.copy(classes)  # Copy to avoid modifying the original array

    # Apply median filter once to the entire temp map
    filtered_size5 = median_filter(classes, size=5)
    filtered_size3 = median_filter(classes, size=3)

    # FIRST: smooth class 3 objects (existing CI)
    class_3_mask = (classes == 3)
    s = generate_binary_structure(2,2) ## eight-connected regions method
    labeled_array, num_features = label(class_3_mask, structure=s)

    for region_label in range(1, num_features + 1):
        region_mask = (labeled_array == region_label)
        region_size = np.sum(region_mask)

        if region_size >= 16:
            # Smooth (median filter with size 5) if the region is slightly larger than the minimum size
            cleaned_map[region_mask] = filtered_size5[region_mask]
        else:
            # Otherwise, leave the region untouched (keep original class 3 values)
            cleaned_map[region_mask] = classes[region_mask]

    # SECOND: smooth class 0 objects (new CI)
    class_0_mask = (classes == 0)
    s = generate_binary_structure(2,2) ## eight-connected regions method
    labeled_array, num_features = label(class_0_mask, structure=s)

    for region_label in range(1, num_features + 1):
        region_mask = (labeled_array == region_label)
        region_size = np.sum(region_mask)

        if region_size >= 16:
            # Smooth (median filter with size 3) if the region is slightly larger than the minimum size
            cleaned_map[region_mask] = filtered_size3[region_mask]
        else:
            # Otherwise, leave the region untouched (keep original class 0 values)
            cleaned_map[region_mask] = classes[region_mask]

    return cleaned_map
    
def get_area_threshold(classes):

    ## get objects which have at least 10 CI pixels (10km^2) by the eight-connected regions method
    ## this will be considered a new CI event for all of those pixels
    ## an existing event will be based on the previous time criteria for all of the CI pixels identified here
    
    CI_events_label = 0 # Classification to check (CI)
    new_class = 2  # New classification for isolated pixels (replace class 0 with class 2)
    neighbor_threshold = 10 # Minimum number of pixels required (10km^2)

    modified_map = classes

    CI_mask = (classes == CI_events_label)

    # make diagonal structure to count objects
    s = generate_binary_structure(2,2)
    labeled_array, num_features = label(CI_mask, structure=s)

    for region_label in range(1, num_features + 1):
        region_mask = (labeled_array == region_label)
        region_size = np.sum(region_mask)

        if region_size >= neighbor_threshold:
            # keep CI event if it has an area larger than 10 pixels (10 km^2)
            modified_map[region_mask] = classes[region_mask]
        else:
            # Otherwise, remove CI events from these pixels
            # Replace these pixels with high clouds (class 2)
            modified_map[region_mask] = new_class
             
    return modified_map

def make_classifications(lon_mrms, lat_mrms, lon_GOES, lat_GOES, refc_timeseries, GOES_timeseries, lon0, lat0, patch_width):

    CI_events_label = 0
    
    for i in range(7):

        refc = refc_timeseries[i,:,:]
        GOES_BT = GOES_timeseries[i,:,:]

        mrms_cropped, GOES_cropped, lon_cropped_mrms, lat_cropped_mrms = get_MRMS_GOES_regrid(lon_mrms, lat_mrms, lon_GOES[i,:,:], lat_GOES[i,:,:], refc, GOES_BT, lon0, lat0, patch_width)
        
        mrms_regridded = median_filter(mrms_cropped, size=3) ## smooth full MRMS REFC field with size 3 filter (not too large to avoid removing extremes)
        GOES_cropped = median_filter(GOES_cropped, size=5) ## smooth full GOES CH13 field with size 5 filter
        
        mrms_zero_regridded = np.where(mrms_regridded < 0, 0, mrms_regridded) ## make -99 values in REFC field == 0 dBZ. Patches with -999 values (bad data) are filtered out in other workflows
        mrms_zero_regridded = np.nan_to_num(mrms_zero_regridded) ## shouldn't be many (or any) nans in the REFC grids acquired but if there are, make 0 dBZ

        binary_GOES_MRMS_labels = np.where(mrms_zero_regridded >= 35, 0, GOES_cropped) ## assign CI category
        
        binary_GOES_MRMS_labels = get_area_threshold(binary_GOES_MRMS_labels)
        
        binary_GOES_MRMS_labels = np.where((GOES_cropped <= 275) & (mrms_zero_regridded < 35), 2, binary_GOES_MRMS_labels) ## assign high clouds/anvil
        binary_GOES_MRMS_labels = np.where((GOES_cropped > 275) & (mrms_zero_regridded < 35), 1, binary_GOES_MRMS_labels) ## assign clear

        if i == 0:

            refc_cropped_current = mrms_zero_regridded
            GOES_cropped_current = GOES_cropped
            
            classifications_full = binary_GOES_MRMS_labels        
            classifications_full = classifications_full.reshape(1, binary_GOES_MRMS_labels.shape[0], binary_GOES_MRMS_labels.shape[1])

            lon_cropped_full = lon_cropped_mrms
            lon_cropped_full = lon_cropped_full.reshape(1, lon_cropped_mrms.shape[0], lon_cropped_mrms.shape[1])

            lat_cropped_full = lat_cropped_mrms
            lat_cropped_full = lat_cropped_full.reshape(1, lat_cropped_mrms.shape[0], lat_cropped_mrms.shape[1])
            
        else:

            binary_GOES_MRMS_labels = binary_GOES_MRMS_labels.reshape(1, binary_GOES_MRMS_labels.shape[0], binary_GOES_MRMS_labels.shape[1])
            classifications_full = np.vstack((classifications_full, binary_GOES_MRMS_labels))

            lon_cropped_mrms = lon_cropped_mrms.reshape(1, lon_cropped_mrms.shape[0], lon_cropped_mrms.shape[1])
            lon_cropped_full = np.vstack((lon_cropped_full, lon_cropped_mrms))

            lat_cropped_mrms = lat_cropped_mrms.reshape(1, lat_cropped_mrms.shape[0], lat_cropped_mrms.shape[1])
            lat_cropped_full = np.vstack((lat_cropped_full, lat_cropped_mrms))

    lon_for_current_plot = lon_cropped_full[0,:,:]
    lat_for_current_plot = lat_cropped_full[0,:,:]

    ## information for time=0 (classification time)
    classifications_final = classifications_full[0,:,:]
    indices_time2 = np.column_stack(np.where(classifications_final == CI_events_label))
    coords_time2 = np.column_stack((lat_for_current_plot[indices_time2[:, 0], indices_time2[:, 1]], 
                                     lon_for_current_plot[indices_time2[:, 0], indices_time2[:, 1]]))

    # create classifications by querying the surrounding space and time of previous frames
    for i in range(6):

        classifications_current = classifications_full[i+1,:,:]
        
        # Extract grid points of interest at t1 and t2
        indices_time1 = np.column_stack(np.where(classifications_current == CI_events_label))
        
        # Get lat/lon for these points
        coords_time1 = np.column_stack((lat_cropped_full[i+1,:,:][indices_time1[:, 0], indices_time1[:, 1]], 
                                     lon_cropped_full[i+1,:,:][indices_time1[:, 0], indices_time1[:, 1]]))

        tree = cKDTree(coords_time1)
        
        # Find the nearest point in t1 (previous time) for each point in t2 (current time)
        _, nearest_indices = tree.query(coords_time2, k=1)

        if len(indices_time1) != 0 and len(indices_time2) != 0:
            
            for j, idx_t1 in enumerate(nearest_indices):
     
                lat1, lon1 = coords_time1[idx_t1]
                lat2, lon2 = coords_time2[j]
            
                coord1 = (lat1, lon1)
                coord2 = (lat2, lon2)
            
                distance_km = distance.distance(coord1, coord2).km
    
                if i < 2:
                    if distance_km <= 15:
                        x2, y2 = indices_time2[j]
                        classifications_final[x2, y2] = 3
                if i >= 2:
                    if distance_km <= 5:
                        x2, y2 = indices_time2[j]
                        classifications_final[x2, y2] = 3
    
    return lon_for_current_plot, lat_for_current_plot, classifications_final, GOES_cropped_current, refc_cropped_current

## calculate and visualize the classifications over the desired region
def plot_classifications(start_date, periods_to_run, lat0, lon0, patch_width, save_path):

    CI_times = pd.date_range(start=start_date, periods=periods_to_run, freq="5min")

    for i in range(len(CI_times)):
    
        CI_time = CI_times[i]
        
        lon_mrms, lat_mrms, refc_timeseries = grab_MRMS_timeseries(CI_time)
        lon_goes, lat_goes, GOES_timeseries = get_GOES_timeseries(CI_time)
        
        lon_cropped, lat_cropped, binary_classifications, GOES_cropped_current, refc_cropped_current = make_classifications(lon_mrms, lat_mrms, lon_goes, lat_goes, refc_timeseries, GOES_timeseries, lon0, lat0, patch_width)
    
        binary_classifications_processed = image_processing(binary_classifications)
    
        #reader = shpreader.Reader('../Shapefiles/ne_10m_admin_2_counties.shp')
        #counties = list(reader.geometries())
        #COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
        
        norm, cmap = ctables.registry.get_with_range("WVCIMSS_r", 200, 350)
    
        rain_colors = ['#FFFFFF','#00F747','#02DB41','#01B636','#038B2A','#F5FF1E','#FFD31E','#FFAA1E','#FF691E','#FF1E1E','#BE2020','#FF28DB']
        rain_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]
        rain_cmap = matplotlib.colors.ListedColormap(rain_colors)
        rain_norm = matplotlib.colors.BoundaryNorm(rain_levels,len(rain_levels))
        
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(131, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(132, projection=ccrs.PlateCarree())
        ax3 = fig.add_subplot(133, projection=ccrs.PlateCarree())

        lon_min = lon0 - (patch_width/2)
        lon_max = lon0 + (patch_width/2)
        lat_min = lat0 - (patch_width/2)
        lat_max = lat0 + (patch_width/2)
    
        ax.set_extent([lon_min+0.25,lon_max-0.25,lat_min+0.25,lat_max-0.25], crs=ccrs.PlateCarree())
        ax2.set_extent([lon_min+0.25,lon_max-0.25,lat_min+0.25,lat_max-0.25], crs=ccrs.PlateCarree())
        ax3.set_extent([lon_min+0.25,lon_max-0.25,lat_min+0.25,lat_max-0.25], crs=ccrs.PlateCarree())
        
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
        
        ## order: CI, MC, HC, EC
        binary_colors = ["#EE80BB", "#D1D1D1", "#6CA4CB", "#A12267"]
        binary_cmap = matplotlib.colors.ListedColormap(binary_colors)
            
        contf = ax.pcolormesh(lon_cropped, lat_cropped, binary_classifications_processed, vmin=0, vmax=3, cmap=binary_cmap, zorder=1, transform=ccrs.PlateCarree())
        contf_GOES = ax2.pcolormesh(lon_cropped, lat_cropped, GOES_cropped_current, vmin=200, vmax=350, cmap=cmap, zorder=1, transform=ccrs.PlateCarree())
        contf_MRMS = ax3.pcolormesh(lon_cropped, lat_cropped, np.ma.masked_array(refc_cropped_current, mask=(refc_cropped_current==0)), cmap=rain_cmap, norm=rain_norm, zorder=1, transform=ccrs.PlateCarree())
    
        upper = 4
        lower = 1
        N = 4
        
        mapper = matplotlib.cm.ScalarMappable(cmap=binary_cmap)
        deltac = (upper-lower)/(2*(N-1))
        mapper.set_array([lower-deltac,upper+deltac])
        
        cax = inset_axes(ax, width='100%', height='3.25%', loc='lower left',
                    bbox_to_anchor=(0,-0.045,1,1), bbox_transform=ax.transAxes, borderpad=0)
        
        cbar = plt.colorbar(mapper, orientation='horizontal', extendfrac=0, shrink=0.19, cax=cax)
        cbar.set_ticks([1,2,3,4])
        cbar.set_ticklabels(['New CI', 'Minimal\nClouds', 'High\nClouds', 'Existing\nConvection'], fontsize=12)
    
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
        
        ax.set_title('CI Classifications\n'+r"$\bf{Valid}$"+": "+str(CI_time)[11:16]+'z '+pd.to_datetime(CI_time).strftime('%b %-d, %Y'), loc='center', fontsize=14, pad=8.5)
        
        ax2.set_title('Channel 13 BT (K)\n'+r"$\bf{Valid}$"+": "+str(CI_time)[11:16]+'z '+pd.to_datetime(CI_time).strftime('%b %-d, %Y'), loc='center', fontsize=14, pad=8.5)
    
        ax3.set_title('REFC (dBZ)\n'+r"$\bf{Valid}$"+": "+str(CI_time)[11:16]+'z '+pd.to_datetime(CI_time).strftime('%b %-d, %Y'), loc='center', fontsize=14, pad=8.5)
    
        plt.tight_layout(w_pad=2.5)
        
        plt.savefig(save_path+'ci_classifications_frame'+str(i)+'.png', bbox_inches='tight', pad_inches=0.25, facecolor='white')

####### EDIT BELOW THIS LINE

start_date = "2024-05-06 19:45" ## start predictions at this time (inclusive)
periods_to_run = 1 ## 5min intervals to run out from the start date
save_path = '../figures/' ## save plots here

lat0 = 37.9 ## central lat for patch
lon0 = -99.5 ## central lon for patch

patch_width = 3
    
plot_classifications(start_date, periods_to_run, lat0, lon0, patch_width, save_path)
