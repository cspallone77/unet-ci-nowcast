from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import numpy as np
import xarray as xr

def reliability_diagram(y_true_expand1, y_true_expand2, y_true_expand3, y_true_expand4, y_true_expand5, probs, bins, save_path, labels, colors, leadtime, class_to_plot):
    """
    Construct a reliability diagram (attribute diagram) for probabilistic forecasts.

    Parameters:
        y_true (array-like): Binary ground truth labels (0 or 1).
        y_pred (array-like): Forecast probabilities (0 to 1).
        bins (int): Number of probability bins.

    Returns:
        None (saves plot)
    """
    
    # Define bin edges and centers
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign each prediction to a bin
    # NOTE: Using right=False, so bins are [a, b). If values==1 exist in your probabilities, consider adding those to the last bin to make more robust.
    # In the U-Net dataset for EC and CI probabilities, no probabilities are exactly 1.0, so this does not affect results.
    bin_indices = np.digitize(probs, bin_edges, right=False) - 1

    # Compute observed frequency per bin (first tolerance)
    observed_freq_expand1 = np.array([
        np.mean(y_true_expand1[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan
        for i in range(bins)
    ])

    # Compute observed frequency per bin (second tolerance)
    observed_freq_expand2 = np.array([
        np.mean(y_true_expand2[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan
        for i in range(bins)
    ])

    # Compute observed frequency per bin (third tolerance)
    observed_freq_expand3 = np.array([
        np.mean(y_true_expand3[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan
        for i in range(bins)
    ])

    # Compute observed frequency per bin (fourth tolerance)
    observed_freq_expand4 = np.array([
        np.mean(y_true_expand4[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan
        for i in range(bins)
    ])

    # Compute observed frequency per bin (fifth tolerance)
    observed_freq_expand5 = np.array([
        np.mean(y_true_expand5[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan
        for i in range(bins)
    ])

    ## count number of predictions per bin
    prediction_counts = np.bincount(bin_indices, minlength=bins)

    # Plot the reliability diagram
    fig = plt.figure(figsize=(15, 12.5))
    ax = fig.add_subplot(111)

    y_true_climo = y_true_expand2 ## use tolerance near the middle of the options as the climo frequency
    ## compute climo event frequency from gridded training labels (taken from one of the tolerances)
    climo = np.mean(y_true_climo) 

    x_values = np.linspace(0, 1, 100)
    no_skill = (x_values + climo) / 2

    ax.plot([0, 1], [0, 1], 'k', linewidth=3)
    ax.plot([0, 1], [climo, climo], 'k--', linewidth=1.5)
    ax.plot([climo, climo], [0, 1], 'k--', linewidth=1.5)
    ax.plot(x_values, no_skill, 'k--', linewidth=1.5)

    ## plot region of skill (gray)
    plt.fill_between(x_values, no_skill, where=(x_values <= climo), color='lightgray', alpha=0.35)
    plt.fill_between(x_values, no_skill, 1, where=(x_values >= climo), color='lightgray', alpha=0.35)

    ## optional: remove bins from plotting that have a very low/unstable prediction count (less than 10, which is the minimum on the plotted inset histogram)
    #observed_freq_expand1 = np.where(prediction_counts < 10, np.nan, observed_freq_expand1)
    #observed_freq_expand2 = np.where(prediction_counts < 10, np.nan, observed_freq_expand2)
    #observed_freq_expand3 = np.where(prediction_counts < 10, np.nan, observed_freq_expand3)
    #observed_freq_expand4 = np.where(prediction_counts < 10, np.nan, observed_freq_expand4)
    #observed_freq_expand5 = np.where(prediction_counts < 10, np.nan, observed_freq_expand5)

    ## plot each bin
    ax.plot(bin_centers, observed_freq_expand1, 'o-', markersize=6.5, linewidth=4, color=colors[0], label=labels[0])
    ax.plot(bin_centers, observed_freq_expand2, 'o-', markersize=6.5, linewidth=4, color=colors[1], label=labels[1])
    ax.plot(bin_centers, observed_freq_expand3, 'o-', markersize=6.5, linewidth=4, color=colors[2], label=labels[2])
    ax.plot(bin_centers, observed_freq_expand4, 'o-', markersize=6.5, linewidth=4, color=colors[3], label=labels[3])
    ax.plot(bin_centers, observed_freq_expand5, 'o-', markersize=6.5, linewidth=4, color=colors[4], label=labels[4])

    ax.legend(loc=9, edgecolor='black', facecolor='whitesmoke', framealpha=1, ncol=3, borderpad=0.5, fontsize=17)
    plt.xlabel('Forecast Probability', fontsize=16, labelpad=18)
    plt.ylabel('Conditional Event Frequency', fontsize=16, labelpad=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.grid(linestyle='--', linewidth=0.6, color='gray')

    inset_left = 0.925
    inset_bottom = 0.55
    inset_width = 0.3
    inset_height = 0.3

    ax_inset = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])

    ax_inset.bar(bin_centers, prediction_counts, edgecolor='black', width=0.05, color='cadetblue')

    ax_inset.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax_inset.yaxis.set_ticks_position("right")
    ax_inset.set_title("Prediction Frequency Per Bin", fontsize=18, pad=11.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax_inset.set_yscale('log')
    ax_inset.set_ylim([1e1, 1e9])
    plt.grid(linestyle='--', linewidth=0.6, color='gray')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    ax.set_title("Reliability Diagram Tolerance Permutations for "+leadtime+" U-Net "+class_to_plot+" Predictions", fontsize=18, pad=13.5)
    plt.grid(linestyle='--', linewidth=0.6, color='gray')

    plt.savefig(save_path+'reliability_permutations_'+leadtime+'_'+class_to_plot+'.png', pad_inches=0.25, facecolor='w', bbox_inches='tight')

def get_binary_labels_preds(labels, probs, category):
    
    all_truth_labels = labels
    y_binary = (all_truth_labels == category).astype(int)
    probs_category = probs[:, category, :, :]
    
    return probs_category, y_binary

## processes data and produces tolerances specified in the labels array below...
def preprocess_reliability_data(y, probs, all_nearby_labels, class_to_plot):

    if class_to_plot == 'CI':
        class_idx = 0
    if class_to_plot == 'EC':
        class_idx = 3
    
    all_nearby_labels_full = all_nearby_labels
    
    all_nearby_labels_30mins = all_nearby_labels_full[:,6:19,:,:]
    all_nearby_labels_15mins = all_nearby_labels_full[:,9:16,:,:]
    all_nearby_labels_5mins = all_nearby_labels_full[:,11:14,:,:]
    
    y_5min = y.copy()
    y_15min = y.copy()
    y_30min = y.copy()
    
    n_samples, n_leadtimes, height, width = all_nearby_labels_30mins.shape
    extra_class = np.any(all_nearby_labels_30mins == class_idx, axis=1) ## 0 for CI
    y_30min[extra_class] = class_idx
    
    n_samples, n_leadtimes, height, width = all_nearby_labels_15mins.shape
    extra_class = np.any(all_nearby_labels_15mins == class_idx, axis=1) ## 0 for CI
    y_15min[extra_class] = class_idx 
    
    n_samples, n_leadtimes, height, width = all_nearby_labels_5mins.shape
    extra_class = np.any(all_nearby_labels_5mins == class_idx, axis=1) ## 0 for CI
    y_5min[extra_class] = class_idx
    
    #probs_category, y_CI_binary_tolerance_0min = get_binary_labels_preds(y, probs, class_idx) ## 0min tolerance, CI class (0)
    probs_category, y_CI_binary_tolerance_5min = get_binary_labels_preds(y_5min, probs, class_idx) ## 5min tolerance, CI class (0)
    probs_category, y_CI_binary_tolerance_15min = get_binary_labels_preds(y_15min, probs, class_idx) ## 15min tolerance, CI class (0)
    probs_category, y_CI_binary_tolerance_30min = get_binary_labels_preds(y_30min, probs, class_idx) ## 30min tolerance, CI class (0)
    
    y_neighbor_1 = maximum_filter(y_CI_binary_tolerance_5min, size=(1, 17, 17), mode='nearest') ## 8km/5min tolerance
    y_neighbor_2 = maximum_filter(y_CI_binary_tolerance_15min, size=(1, 17, 17), mode='nearest') ## 8km/15min tolerance
    y_neighbor_3 = maximum_filter(y_CI_binary_tolerance_30min, size=(1, 17, 17), mode='nearest') ## 8km/30min tolerance
    y_neighbor_4 = maximum_filter(y_CI_binary_tolerance_15min, size=(1, 33, 33), mode='nearest') ## 16km/15min tolerance
    y_neighbor_5 = maximum_filter(y_CI_binary_tolerance_30min, size=(1, 33, 33), mode='nearest') ## 16km/30min tolerance

    return probs_category, y_neighbor_1, y_neighbor_2, y_neighbor_3, y_neighbor_4, y_neighbor_5

all_nearby_labels = np.load('/storage/group/sjg213/default/cps5892/U-Net/testing_files_full/all_nearby_labels.npy')
ds_labels = xr.open_dataset('/storage/group/sjg213/default/cps5892/U-Net/testing_files_full/all_test_labels.nc')
y = ds_labels.labels.values
probs = np.load('/storage/group/sjg213/default/cps5892/U-Net/testing_files_full/10min/all_probs_test_10min.npy')

nan_indices = np.load('/storage/group/sjg213/default/cps5892/U-Net/testing_files_full/10min/nan_indices_test_10min.npy')
y = np.delete(y, nan_indices, axis=0)
all_nearby_labels = np.delete(all_nearby_labels, nan_indices, axis=0)

labels = ["8km & 5min", "8km & 15min", "8km & 30min", "16km & 15min", "16km & 30min"] ## labels for tolerance legend
colors = ["mediumseagreen", "palevioletred", "darkmagenta", "slateblue", "deepskyblue"] ## colors for each tolerance
save_path = '../figures/' ## save plots here
leadtime = '10min' ## adjust leadtime (10min, 20min, 30min, or 40min)
class_to_plot = 'CI' ## class (CI or EC)

## preprocess data
probs_category, y_neighbor_1, y_neighbor_2, y_neighbor_3, y_neighbor_4, y_neighbor_5 = preprocess_reliability_data(y, probs, all_nearby_labels, class_to_plot)

## make diagram
reliability_diagram(y_neighbor_1.flatten(), y_neighbor_2.flatten(), y_neighbor_3.flatten(), y_neighbor_4.flatten(), y_neighbor_5.flatten(), probs_category.flatten(), 15, save_path, labels, colors, leadtime, class_to_plot)

