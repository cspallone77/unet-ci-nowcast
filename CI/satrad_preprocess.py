import s3fs
import tempfile
import gzip
from datetime import datetime, timedelta
from scipy.interpolate import griddata, RegularGridInterpolator, LinearNDInterpolator
from scipy import interpolate 
from goes2go import GOES
import numpy as np
import xarray as xr
from scipy.spatial import Delaunay

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
