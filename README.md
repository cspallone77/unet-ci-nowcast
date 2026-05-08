# unet-ci-nowcast
Assessment of a Multiclass U-Net Segmentation Model for Probabilistic Convective Initiation Nowcasting

# Description

# Repository Structure

- `unet-ci-nowcast/CI/`: This directory contains several scripts for training, model creation, evaluation, and inference for the mutliclass U-Net system.
    - `case_study_analysis.py`: produces the 6x6 panel plots that display a timeseries (1hr 15min long) of U-Net predictions for a case. Inputs allow you to change the starting date and patch location over the Great Plains. The plots visualize the following fields from the top row to the bottom: U-Net 10-min CI, 20-min CI, 20-min EC, nearest HRRR initialization REFC, ground-truth MRMS REFC, and CH13 BT from GOES-16. 
    - `classifications.py`: produces examples of the gridded multiclass labels used in training. Options allow changes to the start date, 5-min periods to run out from the specified time, a central latitude, central longitude, and the patch width (in deg, smaller will run faster).
    - `climatology.py`: functions for making climo plots of the training or testing dataset labels (from an xarray dataset). Plots include a histogram of the CI object areas, histogram of the time of events by UTC hour, and a spatial map of sample counts over the Great Plains in 0.5 deg bins (with options in the function to plot the map for a single month). 
    - `eval_plotting.py`: functions to produce ROC AUC curves and performance diagrams on the testing dataset.
    - `evaluation.py`: code to compute neighborhood max CSI values for the U-Net model and save the results over 10-40mins of leadtime to CSV files. Functions also produce a 95% confidence interval on the max CSI values and save the upper and lower bound of this to CSV.
    - `generate_probs.py`: simple example script to load U-Net model class correctly, load test data, load scaler, and run the model on this data with results saved to a numpy array. 
    - `inference.py`: python script implementation of the run_unet.ipynb notebook found in the notebooks directory. See the README in that folder for more details. 
    - `model.py`: contains the CI_unet model class. This is loaded when the model is used in inference with the specified config (.yml) file settings needed to fill in the inputs. 
    - `reliability.py`: function to process labels and U-Net predictions to produce a neighborhood reliability diagram. The function includes specific tolerances that will be plotted. These tolerances were used for the CI class, but can be changed in the preprocess_reliability_data() function if desired by adjusting the current distance/time tolerance construction. Options allow for changing of the colors of the plot lines, legend labels, the model leadtime to run, and the model class to plot (only CI or EC supported).  
    - `satrad_preprocess.py`: helper functions used to process much of the GOES and MRMS data that is plotted or used in inference. This includes a function for downloading satellite (GOES-16) and radar (MRMS) data from the nearest available time and parallax correcting the GOES-16 data.
    - `trainer.py`: script used for training of the multiclass U-Net model in PyTorch.
    
# Requirements

Download the following packages to run this code:

- Python 3.x 
- PyTorch
- Xarray
- SciPy
- NumPy 
- scikit-learn
- Pandas

# How to Run 

# References

TBD
