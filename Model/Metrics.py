from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology.binary import binary_dilation
from skimage.metrics import structural_similarity as ssim
from Preprocessing.Preprocessor import denoise_in
from skimage.filters import threshold_sauvola
import numpy as np


def sauvola_mask(array, multiplier=255):
    blurred_array = array * multiplier
    blurred_array = blurred_array.astype(int)
    blurred_array_2 = []
    for i in range(128):
        blurred_array_2.append(denoise_in(blurred_array[i], True))
    blurred_array_2 = np.array(blurred_array_2)
    blurred_array_2 = gaussian_filter(blurred_array_2, sigma=2)
    sauvola = threshold_sauvola(blurred_array_2, window_size=7)
    mask_array = blurred_array_2 > sauvola
    mask_array[blurred_array_2 < 20] = False
    mask_array[blurred_array_2 > 200] = True

    for j in range(4):
        mask_array = binary_dilation(mask_array)
        mask_array = binary_fill_holes(mask_array)
        for i in range(128):
            mask_array[i] = binary_fill_holes(mask_array[i])

    return mask_array


def dice(pred, true, k=1):
    intersection = np.sum(pred[true == k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


def masked_ssim_singlechan(prediction_vox, prediction_pix, truth):
    mask_truth = sauvola_mask(truth)
    mask_pix_prediction = sauvola_mask(prediction_pix)
    mask_vox_prediction = sauvola_mask(prediction_vox)
    vox_masked = np.where(mask_vox_prediction, prediction_vox, 0)
    pix_masked = np.where(mask_pix_prediction, prediction_pix, 0)
    real_masked = np.where(mask_truth, truth, 0)

    pix_dice = dice(mask_pix_prediction, mask_truth)
    vox_dice = dice(mask_vox_prediction, mask_truth)
    vox_ssim = ssim(vox_masked, real_masked, win_size=7, channel_axis=0)
    pix_ssim = ssim(pix_masked, real_masked, win_size=7, channel_axis=0)
    return vox_ssim, vox_dice, pix_ssim, pix_dice


def masked_ssim(prediction_vox, prediction_pix, truth):
    mask_truth = sauvola_mask(truth[0] + truth[1])
    mask_pix_prediction = sauvola_mask(prediction_pix[0] + prediction_pix[1])
    mask_vox_prediction = sauvola_mask(prediction_vox[0] + prediction_vox[1])
    vox_masked = np.where(np.array([mask_vox_prediction, mask_vox_prediction]), prediction_vox, 0)
    pix_masked = np.where(np.array([mask_pix_prediction, mask_pix_prediction]), prediction_pix, 0)
    real_masked = np.where(np.array([mask_truth, mask_truth]), truth, 0)

    pix_dice = dice(mask_pix_prediction, mask_truth)
    vox_dice = dice(mask_vox_prediction, mask_truth)
    vox_ssim = ssim(vox_masked, real_masked, win_size=7, channel_axis=0)
    pix_ssim = ssim(pix_masked, real_masked, win_size=7, channel_axis=0)
    return vox_ssim, vox_dice, pix_ssim, pix_dice
