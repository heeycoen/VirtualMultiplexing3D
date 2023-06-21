import numpy as np
from Metrics.dice_scoring import dice
from Metrics.sauvola_masking import sauvola_mask
from skimage.metrics import structural_similarity as ssim


def Masked_SSIM_SingleChan(PredictionVox, PredictionPix, Truth):
    mask_truth = sauvola_mask(Truth)
    mask_pix_prediction = sauvola_mask(PredictionPix)
    mask_vox_prediction = sauvola_mask(PredictionVox)
    VoxMasked = np.where(mask_vox_prediction, PredictionVox, 0)
    PixMasked = np.where(mask_pix_prediction, PredictionPix, 0)
    RealMasked = np.where(mask_truth, Truth, 0)

    PixDice = dice(mask_pix_prediction, mask_truth)
    VoxDice = dice(mask_vox_prediction, mask_truth)
    VoxSSIM = ssim(VoxMasked, RealMasked, win_size=7, channel_axis=0)
    PixSSIM = ssim(PixMasked, RealMasked, win_size=7, channel_axis=0)
    return VoxSSIM, VoxDice, PixSSIM, PixDice


def Masked_SSIM_Double(PredictionVox, PredictionPix, Truth):
    mask_truth = sauvola_mask(Truth[0] + Truth[1])
    mask_pix_prediction = sauvola_mask(PredictionPix[0] + PredictionPix[1])
    mask_vox_prediction = sauvola_mask(PredictionVox[0] + PredictionVox[1])
    VoxMasked = np.where(np.array([mask_vox_prediction, mask_vox_prediction]), PredictionVox, 0)
    PixMasked = np.where(np.array([mask_pix_prediction, mask_pix_prediction]), PredictionPix, 0)
    RealMasked = np.where(np.array([mask_truth, mask_truth]), Truth, 0)

    PixDice = dice(mask_pix_prediction, mask_truth)
    VoxDice = dice(mask_vox_prediction, mask_truth)
    VoxSSIM = ssim(VoxMasked, RealMasked, win_size=7, channel_axis=0)
    PixSSIM = ssim(PixMasked, RealMasked, win_size=7, channel_axis=0)
    return VoxSSIM, VoxDice, PixSSIM, PixDice, mask_truth


def Masked_SSIM_AVG(Prediction, Truth):
    mask_truth = sauvola_mask(Truth[0] + Truth[1])
    mask_prediction = sauvola_mask(Prediction[0] + Prediction[1])
    masked_prediction = np.where(np.array([mask_prediction, mask_prediction]), Prediction, 0)
    RealMasked = np.where(np.array([mask_truth, mask_truth]), Truth, 0)
    Dice = dice(mask_prediction, mask_truth)
    SSIM = ssim(masked_prediction, RealMasked, win_size=7, channel_axis=0)
    return SSIM, Dice, mask_truth, mask_prediction


def Masked_SSIM(Prediction, Truth):
    mask_truth = sauvola_mask(Truth)
    mask_prediction = sauvola_mask(Prediction)
    masked_prediction = np.where(mask_prediction, Prediction, 0)
    RealMasked = np.where(mask_truth, Truth, 0)
    Dice = dice(mask_prediction, mask_truth)
    SSIM = ssim(masked_prediction, RealMasked, win_size=7, channel_axis=0)
    return SSIM, Dice, mask_truth, mask_prediction
