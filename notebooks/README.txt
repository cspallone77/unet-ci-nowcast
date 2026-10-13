This notebook, "run_unet.ipynb", contains a streamlined interface for running the U-Net model and visualizing the results of this system. The following are basic instructions about configuring different settings and running the notebook. Note that there are several comments in the notebook itself that may further clarify certain points.

1. The first cell of the notebook contains all of the necessary Python library imports. Make sure each of these is installed before running through the rest of the notebook. When installed, this cell can be run individually, and each library should load successfully. 
2. Note that "from model import CI_unet" refers to the model.py script (in the scripts folder) that contains the U-Net model class. This script will need to be placed in the same directory as this notebook to ensure this loads successfully.
3. After importing libraries, the second cell of the notebook can be run. This cell contains the major functions that will allow for output to be produced. Functions handle the following tasks: downloading satellite data, radar data, parallax correcting satellite data, regridding the satellite data to the MRMS grid, running the U-Net model using the loaded predictors, and then visualizing the U-Net output in the three panel image.
4. Now that the libraries and functions are loaded a new cell can be made and settings adjusted for a few variables to run the major processes. A few extra cells should already exist underneath the major functions with examples. Feel free to modify these or copy paste the lines in one of these cells to a new cell from which you can run with your own settings.
5. Several settings can be adjusted and then will be passed into the inference() function, which controls the overarching workflow for this notebook. Here is a list of the settings and their role (also commented in the notebook):

leadtime: used to adjust the leadtime of the U-Net model ran. Should be a string taking on the form '10min', '20min', etc.
config_path = path to the folder where the U-Net configs (.yml files) are located as a string. Make sure to include a forward slash at the end to construct the path right. Note that the script requires them by default to have the same filenames as in the configs directory.
weights_path = path to the folder where the U-Net weights are found for each leadtime. Make sure to include a forward slash at the end to construct the path right. The filename for the U-Net weights is taken from the config file "weights_fname" parameter.
scaler_path = path to the folder where the U-Net scalers are found for each leadtime. Make sure to include a forward slash at the end to construct the path right. The filename for the U-Net scalers is taken from the config file "scalers_fname" parameter. 
start_date = start predictions at this time (inclusive). Example: "2024-05-06 18:30". Functions will grab the sat/rad data nearest this time, so it could be any time you wish, but I have mostly used even 5-min intervals for clarity/conciseness on the plot titles. 
periods_to_run = 5min intervals to run out from the start date into the future. 
class_to_plot = the multiclass label to plot predictions from. Options are: 'CI' (convection initiation), 'EC' (existing convection), 'HC' (high cloud), 'MC' (minimal cloud)
save_path = Path to save outputted plots to. Make sure to include a forward slash at the end to construct the path right.

lat0 = central latitude of the patch in deg N (example: 38.65)
lon0 = central longitude of the patch in deg W (example: -99.56)

patch_width = width of the patch (float) to run as a multiple of 128x128 pixels (the training resolution). Example: 2.56 (spacing is every 0.01°, which is ~1km, and produces a grid of 256x256 pixels). I have found this size works well to see a wider area, but running at 128x128 pixels (patch width=1.28) also works well, and larger multiples (ex. 3.84) have run successfully too.

After updating the settings, you can then run that new cell. This should load in as many plots as the number of 5min intervals you specified (in periods_to_run). They should display in the notebook when they all finish, but also save as .png files to the specified output directory as they are made in realtime.

Optional: The Shapefiles folder in this directory contains US county boundaries, which help add some flavor to the back of the map given these plots are usually very zoomed in. There are some options in the plotting functions that can be uncommented that allow for these shapefiles to be plotted if desired.