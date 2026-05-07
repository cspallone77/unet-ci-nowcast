import xarray as xr
import numpy as np
from scipy.ndimage import maximum_filter
from sklearn.metrics import confusion_matrix

def get_all_sample_metrics(y_pred, y_true, y_true_orig, size):
    
    C_arr = np.zeros(y_true.shape[0], dtype=int)
    F_arr = np.zeros(y_true.shape[0], dtype=int)
    M_arr = np.zeros(y_true.shape[0], dtype=int)
    H_arr = np.zeros(y_true.shape[0], dtype=int)
    n = len(C_arr)
    
    for i in range(n):

        y_pred_i = y_pred[i]
        y_true_i = y_true[i]
        y_true_orig_i = y_true_orig[i]

        obs_neighborhood = maximum_filter(y_true_i, size=size, mode='nearest')
        cm_expanded = confusion_matrix(obs_neighborhood.flatten(), y_pred_i.flatten(), labels=[0, 1])
        cm_orig = confusion_matrix(y_true_orig_i.flatten(), y_pred_i.flatten(), labels=[0, 1])
        C_expanded, F_expanded, M_expanded, H_expanded = cm_expanded[0,0], cm_expanded[0,1], cm_expanded[1,0], cm_expanded[1,1]
        C_orig, F_orig, M_orig, H_orig = cm_orig[0,0], cm_orig[0,1], cm_orig[1,0], cm_orig[1,1]

        C_arr[i] = C_orig
        F_arr[i] = F_expanded
        M_arr[i] = M_orig
        H_arr[i] = H_expanded
        
    return C_arr, F_arr, M_arr, H_arr

def bootstrap_from_counts(C_arr, F_arr, M_arr, H_arr, n_bootstrap=1000, seed=77):
    
    np.random.seed(seed)
    n = len(C_arr)
    metrics = []
    
    for _ in range(n_bootstrap):
        
        indices = np.random.choice(n, n, replace=True)
        C_sum = C_arr[indices].sum()
        F_sum = F_arr[indices].sum()
        M_sum = M_arr[indices].sum()
        H_sum = H_arr[indices].sum()
        
        FAR = F_sum / (F_sum + H_sum)
        CSI = H_sum / (H_sum + F_sum + M_sum)
        POD = H_sum / (H_sum + M_sum)

        N = C_sum + F_sum + M_sum + H_sum
        a_r = ((H_sum + F_sum) * (H_sum + M_sum)) / N
        ETS = (H_sum - a_r) / (H_sum + F_sum + M_sum - a_r)
        
        metrics.append([FAR, CSI, POD, ETS])
    
    return np.array(metrics)
    
def binarize_predictions(thresh, probs, y_expanded, y_orig, class_idx):

    all_truth_samples = []
    all_truth_samples_orig = []
    all_preds_samples = []

    for i in range(probs.shape[0]):
                                        
        preds_sample = (probs[i, class_idx, :, :] >= thresh).astype(np.uint8).flatten()
        truth_sample = (y_expanded[i, :, :] == class_idx).astype(np.uint8).flatten()
        truth_sample_orig = (y_orig[i, :, :] == class_idx).astype(np.uint8).flatten()

        all_preds_samples.append(preds_sample)
        all_truth_samples.append(truth_sample)
        all_truth_samples_orig.append(truth_sample_orig)

    all_preds_samples = np.concatenate(all_preds_samples)
    all_truth_samples = np.concatenate(all_truth_samples)
    all_truth_samples_orig = np.concatenate(all_truth_samples_orig)
    
    return all_preds_samples, all_truth_samples, all_truth_samples_orig

def calculate_metrics_threshs(probs, y_expanded, y, size, class_idx):
    
    all_PODs = []
    all_FARs = []
    all_CSIs = []
    all_ETSs = []
    all_lowers = []
    all_uppers = []

    thresholds = np.arange(0,1,0.05) ## thresholds from 0 to 0.95 in steps of 0.05

    for j in range(len(thresholds)):

        thresh = thresholds[j]

        all_preds_samples, all_truth_samples, all_truth_samples_orig = binarize_predictions(thresh, probs, y_expanded, y, class_idx)

        all_preds_samples_2D = all_preds_samples.reshape((probs.shape[0],128,128))
        all_truth_samples_2D = all_truth_samples.reshape((probs.shape[0],128,128))
        all_truth_samples_2D_orig = all_truth_samples_orig.reshape((probs.shape[0],128,128))
        
        C_arr, F_arr, M_arr, H_arr = get_all_sample_metrics(all_preds_samples_2D, all_truth_samples_2D, all_truth_samples_2D_orig, size)
        boot_metrics = bootstrap_from_counts(C_arr, F_arr, M_arr, H_arr)
        
        csi_values = boot_metrics[:, 1]
        lower_conf = np.percentile(csi_values, 2.5) ## 95% conf.
        upper_conf = np.percentile(csi_values, 97.5) ## 95% conf.
        
        all_lowers.append(lower_conf)
        all_uppers.append(upper_conf)
        
        FAR, CSI, POD, ETS = scores_neighborhood(all_truth_samples_2D, all_truth_samples_2D_orig, all_preds_samples_2D, size)
        
        all_PODs.append(POD)
        all_FARs.append(FAR)
        all_CSIs.append(CSI)
        all_ETSs.append(ETS)
    
    all_PODs = np.array(all_PODs)
    all_CSIs = np.array(all_CSIs)
    all_FARs = np.array(all_FARs)
    all_ETSs = np.array(all_ETSs)
    
    max_CSI = np.max(all_CSIs)
    max_ETS = np.max(all_ETSs)
    max_CSI_idx = np.argmax(all_CSIs)
    
    max_POD = all_PODs[max_CSI_idx]
    min_FAR = all_FARs[max_CSI_idx]
    chosen_lower = all_lowers[max_CSI_idx]
    chosen_upper = all_uppers[max_CSI_idx]
    
    return max_CSI, max_POD, min_FAR, max_ETS, chosen_lower, chosen_upper
    
def scores_neighborhood(y_true, y_true_orig, y_pred, size):

    C_all = 0
    F_all = 0
    M_all = 0
    H_all = 0
    
    for i in range(y_true.shape[0]):

        y_pred_i = y_pred[i,:,:]
        y_true_i = y_true[i,:,:]
        y_true_orig_i = y_true_orig[i,:,:]
        
        obs_neighborhood = maximum_filter(y_true_i, size=size, mode='nearest')
        cm_expanded = confusion_matrix(obs_neighborhood.flatten(), y_pred_i.flatten(), labels=[0, 1])
        cm_orig = confusion_matrix(y_true_orig_i.flatten(), y_pred_i.flatten(), labels=[0, 1])
        C_expanded, F_expanded, M_expanded, H_expanded = cm_expanded[0,0], cm_expanded[0,1], cm_expanded[1,0], cm_expanded[1,1]
        C_orig, F_orig, M_orig, H_orig = cm_orig[0,0], cm_orig[0,1], cm_orig[1,0], cm_orig[1,1]
        
        C_all += C_orig
        F_all += F_expanded
        M_all += M_orig
        H_all += H_expanded
        
    FAR = F_all/(F_all+H_all)
    CSI = H_all/(H_all+F_all+M_all)
    POD = H_all/(H_all+M_all)

    n = C_all+F_all+M_all+H_all
    a_r = ((H_all+F_all)*(H_all+M_all))/n
    ETS = (H_all-a_r)/(H_all+F_all+M_all-a_r)
    
    return FAR, CSI, POD, ETS

## evaluate unet predictions using a neighborhood CSI approach over the gridded patch
## return 6 CSV files: Max CSI vals, POD at threshold of max CSI, FAR at threshold of max CSI, Max ETS, lower bound of 95% CSI confidence interval, and upper bound of 95% CSI confidence interval. Each CSV file will have length 4 corresponding to one data point for each leadtime (10-40mins)
## inputs: probs array, labels, surrounding temporal labels, class to predict (4 options), distance tolerance, time tolerance, and path to save outputs to
def evaluate_unet(probs, y, all_nearby_labels, prediction_class, neighborhood_radius, time_tolerance, save_path):
    
    leadtimes = ['10min', '20min', '30min', '40min']

    if prediction_class == 'CI':
        class_idx = 0
    if prediction_class == 'MC':
        class_idx = 1
    if prediction_class == 'HC':
        class_idx = 2
    if prediction_class == 'EC':
        class_idx = 3
        
    ## one value per leadtime...
    all_CSI_vals = np.empty(4)
    all_FAR_vals = np.empty(4)
    all_POD_vals = np.empty(4)
    all_ETS_vals = np.empty(4)
    upper_CIs = np.empty(4)
    lower_CIs = np.empty(4)
    
    for i in range(len(leadtimes)):

        #print(leadtimes[i])
    
        dist_val = (neighborhood_radius*2)+1
        leadtime = leadtimes[i]
        
        ## invoke clean env
        #test_label_np = y
        #if_class_1 = np.sum(test_label_np == 1, axis=(1, 2))
        #total_pixels = test_label_np.shape[1] * test_label_np.shape[2] 
        #mask = if_class_1 > (total_pixels / 2)
        
        #y = y[mask]
        #probs = probs[mask]
        #all_nearby_labels = all_nearby_labels[mask]
        
        y_time_tolerance = y.copy()

        if time_tolerance == '30min':
            all_nearby_labels_time_tolerance = all_nearby_labels[:,6:19,:,:]
        if time_tolerance == '15min':
            all_nearby_labels_time_tolerance = all_nearby_labels[:,9:16,:,:]
        if time_tolerance == '10min':
            all_nearby_labels_time_tolerance = all_nearby_labels[:,10:15,:,:]
        if time_tolerance == '5min':
            all_nearby_labels_time_tolerance = all_nearby_labels[:,11:14,:,:]
        if time_tolerance == '0min':
            pass

        if time_tolerance != '0min':
            n_samples, n_leadtimes, height, width = all_nearby_labels_time_tolerance.shape
            class_masks = np.any(all_nearby_labels_time_tolerance == class_idx, axis=1)
            y_time_tolerance[class_masks] = class_idx
    
        CSI, POD, FAR, ETS, lowerCI, upperCI = calculate_metrics_threshs(probs, y_time_tolerance, y, dist_val, class_idx)
    
        all_CSI_vals[i] = CSI
        all_FAR_vals[i] = FAR
        all_POD_vals[i] = POD
        all_ETS_vals[i] = ETS
        lower_CIs[i] = lowerCI
        upper_CIs[i] = upperCI
            
    np.savetxt(save_path+'CSI_vals_leadtimes_'+prediction_class+'.csv', all_CSI_vals, delimiter=',')
    np.savetxt(save_path+'FAR_vals_leadtimes_'+prediction_class+'.csv', all_FAR_vals, delimiter=',')
    np.savetxt(save_path+'POD_vals_leadtimes_'+prediction_class+'.csv', all_POD_vals, delimiter=',')
    np.savetxt(save_path+'ETS_vals_leadtimes_'+prediction_class+'.csv', all_ETS_vals, delimiter=',')
    np.savetxt(save_path+'all_lower_confs_'+prediction_class+'.csv', lower_CIs, delimiter=',')
    np.savetxt(save_path+'all_upper_confs_'+prediction_class+'.csv', upper_CIs, delimiter=',')
  
#all_nearby_labels = np.load('/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/all_nearby_labels_test.npy')
#ds_labels = xr.open_dataset('/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/all_test_labels.nc')
#y = ds_labels.labels.values
#probs = np.load('/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/10min/probs_test_10min.npy')

#nan_indices = np.load('/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/10min/nan_indices_test_10min.npy')
#y = np.delete(y, nan_indices, axis=0)
#all_nearby_labels = np.delete(all_nearby_labels, nan_indices, axis=0)

#prediction_class = 'CI' ## class to evaluate performance on. Options are 'CI', 'EC', 'HC', and 'MC'
#neighborhood_radius = 16 ## in km, calculation follows Schmidt et al. (2024). Use 1 for no tolerance (original pixels).
#time_tolerance = '30min' ## +/- the event time. Options are 0min, 5min, 10min, 15min, or 30min. 0min means no time tolerance.
#save_path = './animation/' ## path to save files to

#evaluate_unet(probs, y, all_nearby_labels, prediction_class, neighborhood_radius, time_tolerance, save_path)
