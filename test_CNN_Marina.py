import torch
import os
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import nibabel as nib
from sklearn.metrics import f1_score
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader
from data import data_loader
import argparse

import copy

# ===================
# Resize image
# ===================
def image_resize(img_seg, sz2):
    temp = np.zeros((img_seg.shape[0], sz2, img_seg.shape[2]))
    
    for i in range(img_seg.shape[0]):
        temp[i, :, :] = (resize(img_seg[i, :, :], (sz2, img_seg.shape[2])) > 0).astype(np.uint8)
        
    return temp

def match_manual_seg_to_volume(manual_seg, volume):
    manual_seg_volume = image_resize(manual_seg, volume.shape[1]).astype(int)
    _, ind_y, _ = np.where(manual_seg_volume == 1)

    if ind_y.size == 0:
        return manual_seg_volume

    min_ind_y = np.min(ind_y)
    for i in range(max(min_ind_y-10, 0), min_ind_y):
        manual_seg_volume[:, i, :] = manual_seg_volume[:, min_ind_y, :]

    return manual_seg_volume

def plot_dice_vs_bat_volumes(target_bat_volume_arr, pred_bat_volume_arr, dice_arr, path_to_save):
    target_bat_volume_arr = np.array(target_bat_volume_arr)
    pred_bat_volume_arr = np.array(pred_bat_volume_arr)
    dice_arr = np.array(dice_arr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].scatter(target_bat_volume_arr, dice_arr, s=35, alpha=0.8, edgecolors='black', linewidths=0.4)
    axes[0].set_xlabel('Target BAT volume')
    axes[0].set_ylabel('Dice score')
    axes[0].set_title('Dice vs Target BAT volume')

    axes[1].scatter(pred_bat_volume_arr, dice_arr, s=35, alpha=0.8, edgecolors='black', linewidths=0.4)
    axes[1].set_xlabel('Predicted BAT volume')
    axes[1].set_title('Dice vs Predicted BAT volume')

    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(path_to_save, 'dice_vs_bat_volume.png'), dpi=300)
    plt.close(fig)

# ===============
# Perform segmentation using CT and PET thresholds
# ===============
def segmentation_using_threshold(ct_volume, pet_volume, ct_min, ct_max, pet_min = None):
    ct_th1 = ct_volume > ct_min
    ct_th2 = ct_volume < ct_max
    seg = np.logical_and(ct_th1, ct_th2)

    if pet_min != None:
        pet_th = pet_volume > pet_min
        seg = np.logical_and(seg, pet_th)
    
    uncorrected_seg = copy.deepcopy(seg)

    return seg, uncorrected_seg

# ===============
# Denormalize batch using the normalization parameters
# ===============
def denormalize_batch(img, norm_params, img_type):
    img_unnorm = img.clone()
    for i in range(img.shape[0]):
        if img_type == 'CT':
            ct_min, ct_max = norm_params[i].squeeze()[0], norm_params[i].squeeze()[1]
            img_unnorm[i] = img[i] * (ct_max - ct_min) + ct_min
        elif img_type == 'PET':
            pet_min, pet_max = norm_params[i].squeeze()[2], norm_params[i].squeeze()[3]
            img_unnorm[i] = img[i] * (pet_max - pet_min) + pet_min

    return img_unnorm

# ===================
# Calculate SUV stats
# ===================
def calculate_suv_stats(pet_volume, seg_volume_org):
    mask_fp = np.zeros_like(pet_volume)

    mask_fp = 1 - mask_fp
    seg_volume = np.logical_and(mask_fp, seg_volume_org)

    ind_seg = np.where(seg_volume == True)
    pet_roi = pet_volume[ind_seg[0], ind_seg[1], ind_seg[2]]

    if ind_seg[0].size == 0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        bat_volume = pet_roi.shape[0] * 0.0014305100097
        mean_suv = np.mean(pet_roi)
        max_suv = np.max(pet_roi)
        median_suv = np.median(pet_roi)
        percentile_99_suv = np.percentile(pet_roi, 99)
        percentile_95_suv = np.percentile(pet_roi, 95)
        percentile_90_suv = np.percentile(pet_roi, 90)
    
    return bat_volume, mean_suv, max_suv, median_suv, percentile_99_suv, percentile_95_suv, percentile_90_suv

def test_segmentation_network(model, loader_test, manual_seg, device, pet_th, path_to_save_quantitative_results):

    affine = np.eye(4)
    affine[0,0], affine[1,1], affine[2,2], affine[3,3] = -0.9765625, -0.9765625, 1.5, 1.0
    
    # create directory to save results
    os.makedirs(path_to_save_quantitative_results, exist_ok=True)

    # switch to eval mode
    model.eval()
    
    # arrays to store mean/max/median SUV values and dice scores that will be written to csv files.
    target_mean_suv_arr = []
    target_max_suv_arr = []
    target_median_suv_arr = []
    target_bat_volume_arr = []

    pred_mean_suv_arr = []
    pred_max_suv_arr = []
    pred_median_suv_arr = []
    pred_bat_volume_arr = []

    baseline_180_10_mean_suv_arr = []
    baseline_180_10_max_suv_arr = []
    baseline_180_10_median_suv_arr = []
    baseline_180_10_bat_volume_arr = []

    dice_arr = []
    dice_baseline_180_10_arr = []

    # arrays to store ct/pred/seg volumes and their corresponding subject names
    all_volume_names = []

    # a boolean that is used to form 3D images from 2D slices.
    isFirst = True

    for input, target, norm_params, names in loader_test:
        # Read input CT and target PET images, and arrange shape
        input, target = input.to(device), target.to(device)
        input, target = input.squeeze(-1), target.squeeze(-1)

        # Give CT as input to the model and get predicted PET
        pred_logits, _ = model(input)

        # Denormalize input, target and predicted images to reverse the effect of normalization done during training
        # If the data loader provides already unnormalized images, then this step is only needed for the predicted images.
        input_denorm = denormalize_batch(input, norm_params, 'CT')
        target_denorm = denormalize_batch(target, norm_params, 'PET')
        pred_logits_denorm = denormalize_batch(pred_logits, norm_params, 'PET')

        # We get 2D slices from the data loader. Here we form 3D images from those 2D slices.
        name = names[0]

        # Process the first slice and the subsequent slices of a subject differently.
        if isFirst:
            isFirst = False
            volume_name = name
            input_volume = input_denorm.squeeze().unsqueeze(1).detach().cpu().numpy()
            target_volume = target_denorm.squeeze().unsqueeze(1).detach().cpu().numpy()
            pred_logits_volume = pred_logits_denorm.squeeze().unsqueeze(1).detach().cpu().numpy()
        else:
            if volume_name == name:
                input_volume = np.concatenate((input_volume, input_denorm.squeeze().unsqueeze(1).detach().cpu().numpy()), axis = 1)
                target_volume = np.concatenate((target_volume, target_denorm.squeeze().unsqueeze(1).detach().cpu().numpy()), axis = 1)
                pred_logits_volume = np.concatenate((pred_logits_volume, pred_logits_denorm.squeeze().unsqueeze(1).detach().cpu().numpy()), axis = 1)
            else:
                all_volume_names.append(volume_name)
                manual_seg_volume = match_manual_seg_to_volume(manual_seg, input_volume)

                # Baseline segmentation - using only CT thresholding
                _, segmentation_baseline_180_10 = segmentation_using_threshold(input_volume, target_volume, -180, -10, 0)
                segmentation_baseline_180_10 = np.logical_and(manual_seg_volume, segmentation_baseline_180_10)

                # Segment target and predicted PET using PET thresholding
                seg_volume_target = target_volume > pet_th
                seg_volume_pred = pred_logits_volume > pet_th

                # Only take the region within the manual segmentation mask, i.e., supraclavicular region
                seg_volume_target = np.logical_and(manual_seg_volume, seg_volume_target)
                seg_volume_pred = np.logical_and(manual_seg_volume, seg_volume_pred)

                # Calculate Dice score for the network's segmentation and the baseline segmentation
                dice = f1_score(seg_volume_target.reshape(-1), seg_volume_pred.reshape(-1), zero_division=1)
                dice_baseline_180_10 = f1_score(seg_volume_target.reshape(-1), segmentation_baseline_180_10.reshape(-1), zero_division=1)

                # Add the calculated dice scores to the corresponding arrays
                dice_arr.append(dice)
                dice_baseline_180_10_arr.append(dice_baseline_180_10)
                
                # Calculate SUV stats
                stats_target_pet_th = calculate_suv_stats(target_volume, seg_volume_target)
                stats_pred_pet_th = calculate_suv_stats(pred_logits_volume, seg_volume_pred)
                stats_pred_baseline_180_10 = calculate_suv_stats(target_volume, segmentation_baseline_180_10)

                # Append the calculated SUV stats to the corresponding arrays
                target_bat_volume_arr.append(stats_target_pet_th[0]) # bat volume
                target_mean_suv_arr.append(stats_target_pet_th[1]) # mean_suv
                target_max_suv_arr.append(stats_target_pet_th[2]) # max_suv
                target_median_suv_arr.append(stats_target_pet_th[3]) # median_suv

                pred_bat_volume_arr.append(stats_pred_pet_th[0]) # bat volume
                pred_mean_suv_arr.append(stats_pred_pet_th[1]) # mean_suv
                pred_max_suv_arr.append(stats_pred_pet_th[2]) # max_suv
                pred_median_suv_arr.append(stats_pred_pet_th[3]) # median_suv

                baseline_180_10_bat_volume_arr.append(stats_pred_baseline_180_10[0]) # bat volume
                baseline_180_10_mean_suv_arr.append(stats_pred_baseline_180_10[1]) # mean_suv
                baseline_180_10_max_suv_arr.append(stats_pred_baseline_180_10[2]) # max_suv
                baseline_180_10_median_suv_arr.append(stats_pred_baseline_180_10[3]) # median_suv

                # Print only dice scores
                print('%s, %.3f, %.3f'%(volume_name, dice, dice_baseline_180_10))
                print('Target BAT_volume: %.3f, Predicted BAT_volume: %.3f, Baseline BAT_volume_180_10: %.3f'%(
                    stats_target_pet_th[0],
                    stats_pred_pet_th[0],
                    stats_pred_baseline_180_10[0],
                ))
                
                # Proceed with the next subject
                volume_name = name
                input_volume = input_denorm.squeeze().unsqueeze(1).detach().cpu().numpy()
                target_volume = target_denorm.squeeze().unsqueeze(1).detach().cpu().numpy()
                pred_logits_volume = pred_logits_denorm.squeeze().unsqueeze(1).detach().cpu().numpy()

    all_volume_names.append(volume_name)
    manual_seg_volume = match_manual_seg_to_volume(manual_seg, input_volume)

    # Baseline segmentation - using only CT thresholding
    _, segmentation_baseline_180_10 = segmentation_using_threshold(input_volume, target_volume, -180, -10)
    segmentation_baseline_180_10 = np.logical_and(manual_seg_volume, segmentation_baseline_180_10)

    # Segment target and predicted PET using PET thresholding
    seg_volume_target = target_volume > pet_th
    seg_volume_pred = pred_logits_volume > pet_th

     # Only take the region within the manual segmentation mask, i.e., supraclavicular region
    seg_volume_target = np.logical_and(manual_seg_volume, seg_volume_target)
    seg_volume_pred = np.logical_and(manual_seg_volume, seg_volume_pred)
    
    # Calculate Dice score for the network's segmentation and the baseline segmentation
    dice = f1_score(seg_volume_target.reshape(-1), seg_volume_pred.reshape(-1), zero_division=1)
    dice_baseline_180_10 = f1_score(seg_volume_target.reshape(-1), segmentation_baseline_180_10.reshape(-1), zero_division=1)

    # Add the calculated dice scores to the corresponding arrays
    dice_arr.append(dice)
    dice_baseline_180_10_arr.append(dice_baseline_180_10)
    
    # Calculate SUV stats
    stats_target_pet_th = calculate_suv_stats(target_volume, seg_volume_target)
    stats_pred_pet_th = calculate_suv_stats(pred_logits_volume, seg_volume_pred)
    stats_pred_baseline_180_10 = calculate_suv_stats(target_volume, segmentation_baseline_180_10)
    

    # Append the calculated SUV stats to the corresponding arrays
    target_bat_volume_arr.append(stats_target_pet_th[0]) # bat volume
    target_mean_suv_arr.append(stats_target_pet_th[1]) # mean_suv
    target_max_suv_arr.append(stats_target_pet_th[2]) # max_suv
    target_median_suv_arr.append(stats_target_pet_th[3]) # median_suv

    pred_bat_volume_arr.append(stats_pred_pet_th[0]) # bat volume
    pred_mean_suv_arr.append(stats_pred_pet_th[1]) # mean_suv
    pred_max_suv_arr.append(stats_pred_pet_th[2]) # max_suv
    pred_median_suv_arr.append(stats_pred_pet_th[3]) # median_suv

    baseline_180_10_bat_volume_arr.append(stats_pred_baseline_180_10[0]) # bat volume
    baseline_180_10_mean_suv_arr.append(stats_pred_baseline_180_10[1]) # mean_suv
    baseline_180_10_max_suv_arr.append(stats_pred_baseline_180_10[2]) # max_suv
    baseline_180_10_median_suv_arr.append(stats_pred_baseline_180_10[3]) # median_suv

    # Print only dice scores
    print('%s, %.3f, %.3f'%(volume_name, dice, dice_baseline_180_10))
    print('Target BAT_volume: %.3f, Predicted BAT_volume: %.3f'%(
        stats_target_pet_th[0],
        stats_pred_pet_th[0],
    ))
    
    # # =======================
    # # Calculate pearson correlation between suv stats of target and predicted segmentations
    # # =======================
    # pearson_mean_suv, _ = pearsonr(np.array(target_mean_suv_arr).reshape(-1), np.array(pred_mean_suv_arr).reshape(-1))
    # pearson_median_suv, _ = pearsonr(np.array(target_median_suv_arr).reshape(-1), np.array(pred_median_suv_arr).reshape(-1))
    # pearson_max_suv, _ = pearsonr(np.array(target_max_suv_arr).reshape(-1), np.array(pred_max_suv_arr).reshape(-1))

    # =====================================
    # SAVE RESULTS TO A DIFFERENT CSV FILE
    # =====================================
    # All Results
    if os.path.exists('results.csv'):
        df = pd.read_csv('results.csv')
        df['subject_name_fold'] = all_volume_names
        df['Target BAT_volume_fold'] =  target_bat_volume_arr
        df['Predicted BAT_volume_fold'] = pred_bat_volume_arr
        df['Target SUV_mean_fold'] =  target_mean_suv_arr
        df['Predicted SUV_mean_fold'] = pred_mean_suv_arr
        df['Target SUV_median_fold'] =  target_median_suv_arr
        df['Predicted SUV_median_fold'] = pred_median_suv_arr
        df['Target SUV_max_fold'] =  target_max_suv_arr
        df['Predicted SUV_max_fold'] = pred_max_suv_arr
        df['Dice Score_fold'] = dice_arr
        df['Baseline BAT_volume_180_10_fold'] = baseline_180_10_bat_volume_arr
        df['Dice Score_baseline_180_10_fold'] = dice_baseline_180_10_arr

    else:
        df = pd.DataFrame({'subject_name': all_volume_names,
                            'Target BAT_volume':  target_bat_volume_arr,
                            'Predicted BAT_volume': pred_bat_volume_arr,
                            'Target SUV_mean': target_mean_suv_arr,
                            'Predicted SUV_mean': pred_mean_suv_arr,
                            'Target SUV_median': target_median_suv_arr,
                            'Predicted SUV_median': pred_median_suv_arr,
                            'Target SUV_max': target_max_suv_arr,
                            'Predicted SUV_max': pred_max_suv_arr,
                            'Dice Score': dice_arr,
                            'Baseline BAT_volume_180_10': baseline_180_10_bat_volume_arr,
                            'Dice Score_baseline_180_10': dice_baseline_180_10_arr})

    df_bat_volumes = pd.DataFrame({
        'subject_name': all_volume_names,
        'Target BAT_volume': target_bat_volume_arr,
        'Predicted BAT_volume': pred_bat_volume_arr,
        'Baseline BAT_volume_180_10': baseline_180_10_bat_volume_arr,
    })
    df_bat_volumes.to_csv(os.path.join(path_to_save_quantitative_results, 'bat_volumes.csv'), index=False)
    plot_dice_vs_bat_volumes(
        target_bat_volume_arr,
        pred_bat_volume_arr,
        dice_arr,
        path_to_save_quantitative_results,
    )

    df.to_csv('results.csv', index = False)

    print('Dice \t Dice_baseline_180_10')
    print('%.3f \t %.3f'%(np.mean(dice_arr), np.mean(dice_baseline_180_10_arr)))


def main(exp_config):

    # =====================
    # Parameters
    # =====================
    pet_th = 1.5
    path_to_save_quantitative_results = "./results" # TODO: set path to save quantitative results"

    # =====================
    # Define network architecture
    # =====================
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = exp_config.model
    model.to(device)

    # =========================
    # Load pre_trained mode
    # =========================
    path_to_load_pretrained_model = exp_config.path_to_save_pretrained_models + "model.pth" # TODO: set path to load the pre-trained model that I shared with you
    model.load_state_dict(torch.load(path_to_load_pretrained_model, map_location=torch.device(device), weights_only=True))

    # =========================
    # Load source dataset
    # =========================
    _, test_loader, _ = data_loader.load_datasets(exp_config,
                                                    exp_config.batch_size,\
                                                    exp_config.path_train,\
                                                    exp_config.path_test,\
                                                    exp_config.path_validation,\
                                                    exp_config.tf)

    # =========================
    # Load manual segmentation mask
    # =========================
    path = '../data/PT_CT_BAT/segmentation_template.nii.gz' # TODO: replace with the manual segmentation mask path that I shared with you
    manual_seg = np.transpose(nib.load(path).get_fdata(), [0, 2, 1])

    # =========================
    # Test on source data
    # =========================
    test_segmentation_network(model, test_loader, manual_seg, device, pet_th, path_to_save_quantitative_results)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("--config_path", type=str, help="Path to experiment config file", required=True)
    args = parser.parse_args()

    config_file = args.config_path
    config_module = config_file.split('/')[-1].rstrip('.py')
        
    exp_config = SourceFileLoader(config_module, config_file).load_module() # exp_config stores configurations in the given config file under experiments folder.
    main(exp_config=exp_config)
