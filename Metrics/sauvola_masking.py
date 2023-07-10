from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters.thresholding import threshold_sauvola
from skimage.morphology.binary import binary_dilation

import cv2
import numpy as np


def denoise_in(input, denoise):
    if denoise:
        return cv2.fastNlMeansDenoising(input.astype('uint8'), None, 10, 7, 21)
    else:
        return input


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

    for i in range(4):
        mask_array = binary_dilation(mask_array)
        mask_array = binary_fill_holes(mask_array)
        for i in range(128):
            mask_array[i] = binary_fill_holes(mask_array[i])

    return mask_array
