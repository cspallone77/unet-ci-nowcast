import numpy as np
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import confusion_matrix, mean_squared_error, brier_score_loss, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import xarray as xr 

def compute_performance_curve(fname_pod, fname_sr, class_plotted, leadtime, save_path):
    
    # Load POD and SR arrays (previously calculated from evaluation code)
    
    POD = np.loadtxt(fname_pod, delimiter=',')
    SR = np.loadtxt(fname_sr, delimiter=',')

    # Create meshgrid for CSI contours
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    CSI = X * Y / (X + Y - X * Y + 1e-8)  # Avoid division by zero
    CSI = np.nan_to_num(CSI)

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111)

    # Filled CSI contours
    csi_levels = np.arange(0, 1.1, 0.1)
    contf = ax.contourf(X, Y, CSI, levels=csi_levels, cmap="Purples", alpha=0.5)
    cont = ax.contour(X, Y, CSI, levels=csi_levels, colors='black', linewidths=0.35, linestyles='-', zorder=-2)

    # Plot performance curves
    ax.plot(SR, POD, marker='o', markersize=0, linewidth=2.5, color='#EE80BB', label=leadtime+' '+class_plotted, path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()])

    # Compute CSI per point
    def compute_csi(pod, sr):
        return ((pod * sr) / (pod + sr - pod * sr + 1e-8))

    CSI_class = compute_csi(POD, SR)

    # Identify index of max CSI
    max_idx_CSI = np.nanargmax(CSI_class)

    # Get SR, POD, and CSI at max

    max_x = SR[max_idx_CSI]
    max_y = POD[max_idx_CSI]
    max_csi = CSI_class[max_idx_CSI]

    # Plot stars at max CSI
    ax.scatter(max_x, max_y, marker='*', color='#EE80BB', s=250, edgecolors='black', zorder=10)

    bias_vals = [0.25, 0.5, 1.0, 2.0, 4.0]  # Choose what you want to display
    sr_line = np.linspace(0.001, 1.0, 500)

    for fb in bias_vals:
        pod_line = fb * sr_line

        # Mask values that exceed plot bounds
        pod_line[pod_line > 1] = np.nan

        ax.plot(
            sr_line, pod_line,
            linestyle='--', color='black', linewidth=1.4, alpha=0.7, zorder=-1
        )

        # Add labels near the right/top edge
        valid = np.isfinite(pod_line)

        if np.any(valid):
            # choose a point along the visible segment (0.85 fraction)
            idx = int(0.50 * np.sum(valid))
            x_lab = sr_line[valid][idx]
            y_lab = pod_line[valid][idx]

            # pick a small second point offset in data space to compute local slope in display coords
            dx = 0.03  # small step in SR (tweak if your SR range differs)
            x2 = x_lab + dx
            y2 = fb * x2
            # clip second point if outside plot
            if x2 > 1 or y2 > 1:
                dx = -dx
                x2 = x_lab + dx
                y2 = fb * x2

            # transform data coords to display (pixel) coords
            p1 = ax.transData.transform((x_lab, y_lab))
            p2 = ax.transData.transform((x2, y2))

            # angle in degrees in display space
            angle = np.degrees(np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])))

            ax.text(
                x_lab,
                y_lab,
                f'FB={fb}',
                fontsize=10,
                color='black',
                rotation=angle,
                rotation_mode='anchor',   # rotate around the text anchor
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='black', alpha=1),
                transform=ax.transData,   # place in data coords
                clip_on=True,
                alpha=1, zorder=2
            )
    
    # Customize plot
    ax.legend(loc=3, edgecolor='black', facecolor='whitesmoke', framealpha=1, ncol=1, borderpad=0.5, fontsize=14)
    plt.xlabel('Success Ratio (1 - FAR)', fontsize=15, labelpad=15)
    plt.ylabel('Probability of Detection (POD)', fontsize=15, labelpad=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # Colorbar
    cax = inset_axes(ax, width='2.5%', height='100%', loc='lower right',
                     bbox_to_anchor=(.0375, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(contf, orientation='vertical', extendfrac=0, cax=cax)
    cbar.set_ticks(np.arange(0, 1.1, 0.1))
    cax.tick_params(labelsize=15)
    cbar.set_label("Critical Success Index (CSI)", fontsize=15, rotation=270, labelpad=35)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.set_title("U-Net Performance Diagram for "+leadtime+' '+class_plotted+' Predictions', fontsize=18, pad=14)

    plt.savefig(save_path+'performance_diagram_'+class_plotted+'_'+leadtime+'.png', pad_inches=0.25, facecolor='w', bbox_inches='tight')

def plot_roc_curve(truth_labels, probs, class_plotted, leadtime, save_path):

    if class_plotted == 'CI':
        class_idx = 0
    if class_plotted == 'MC':
        class_idx = 1
    if class_plotted == 'HC':
        class_idx = 2
    if class_plotted == 'EC':
        class_idx = 3
    
    fig = plt.figure(figsize=(10,12))
    ax = fig.add_subplot(111)
    
    x = np.arange(0, 1.05, 0.05)
    y = np.arange(0, 1.05, 0.05)
    X, Y = np.meshgrid(x, y)
    PSS_grid = Y - X

    truth_labels = np.where(truth_labels.flatten() == class_idx, 1, 0)
    probs = probs[:,class_idx,:,:].flatten()
   
    # ROC + AUC
    fpr, tpr, _ = roc_curve(truth_labels, probs)
 
    auc_score = roc_auc_score(truth_labels, probs)
  
    # Plot ROC curves
    ax.plot(fpr, tpr, linewidth=4, linestyle='solid', color="#EE80BB", label=f'U-Net (AUC={auc_score:.3f})')

    # Plot background PSS contours
    contf = ax.contourf(X, Y, PSS_grid, levels=np.arange(0, 1.1, 0.1), cmap="Blues", alpha=0.5)
    cont = ax.contour(X, Y, PSS_grid, levels=np.arange(0, 1.1, 0.1), colors='black', linewidths=0.35, linestyles='-', zorder=-2)

    # Annotate max PSS with star
    def annotate_max_pss(fpr, tpr, auc_val, color, label_offset=(0.02, -0.02)):
        pss = tpr - fpr
        max_idx = np.argmax(pss)
        fpr_max, tpr_max = fpr[max_idx], tpr[max_idx]
        ax.plot(fpr_max, tpr_max, marker='*', color=color, markersize=18, zorder=20, markeredgecolor='black')
       
    annotate_max_pss(fpr, tpr, auc_score, "#EE80BB")
   
    cax = inset_axes(ax, width='2.5%', height='100%', loc='lower right',
            bbox_to_anchor=(.0375,0,1,1), bbox_transform=ax.transAxes, borderpad=0)

    cbar = plt.colorbar(contf, orientation='vertical', extendfrac=0, cax=cax)
    cbar.set_ticks(np.arange(0,1.1,0.1))
    cax.tick_params(labelsize=16)
    cbar.set_label("Peirce Skill Score (PSS)", fontsize=16, rotation=270, labelpad=37.5)

    # Add reference line
    ax.plot(x, y, linewidth=2, color='black', linestyle='--', label='Random Chance (AUC=0.500)')

    ax.legend(loc=4, edgecolor='black', facecolor='whitesmoke', borderpad=0.5, framealpha=1, ncol=1, fontsize=19)
    ax.set_xlabel('False Positive Rate', fontsize=16, labelpad=18)
    ax.set_ylabel('True Positive Rate', fontsize=16, labelpad=18)
    ax.set_xticks(np.arange(0,1.1,0.1))
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.tick_params(labelsize=16)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    
    ax.set_title("U-Net ROC AUC for "+leadtime+' '+class_plotted+' Predictions', fontsize=18, pad=13.5)
    
    plt.savefig(save_path+'unet_roc_auc_'+class_plotted+'_'+leadtime+'.png', pad_inches=0.25, facecolor='w', bbox_inches='tight')

#probs = np.load('/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/10min/probs_test_10min.npy') ## load probability values
#ds_labels = xr.open_dataset('/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/all_test_labels.nc') ## load ground truth
#y = ds_labels.labels.values ## labels

#nan_indices = np.load('/storage/group/sjg213/default/cps5892/CS_AI4ES_2026/testing_data/10min/nan_indices_test_10min.npy') ## nans to remove
#y = np.delete(y, nan_indices, axis=0) ## remove nans from labels

#save_path = '../figures/' ## path to save figures
#class_plotted = 'CI' ## class that will be plotted (CI, EC, HC, or MC)
#leadtime = '10min' ## model leadtime

#plot_roc_curve(y, probs, class_plotted, leadtime, save_path) ## plot ROC AUC

#fname_pod = './unet_files/pod_thresholds_CI.csv' ## fname to pod values (1D array, comma delimited CSV)
#fname_sr = './unet_files/sr_thresholds_CI.csv' ## fname to sr values (1D array, comma delimited CSV)

#compute_performance_curve(fname_pod, fname_sr, class_plotted, leadtime, save_path) ## plot performance diagram from pre-calculated POD and SR values over probability thresholds
