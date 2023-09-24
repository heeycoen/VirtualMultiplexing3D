from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
import ssim_scoring
import os
from glob import glob
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error
import yaml
from yaml.loader import SafeLoader
import argparse
from sewar.full_ref import mse, rmse
from ssim_scoring import sauvola_mask, Masked_SSIM_TruthMask, Masked_SSIM_AVG_TruthMask
from pytorch_msssim import ms_ssim
import torch


def compare_models_MS_SSIM(root, names, dataset_names):
    z = [x.split("/") for x in
         glob(root + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    z = [x[0:len(x) - 3] for x in z]
    ssim_keys = ['MSSIM',
                 'NSSIM', 'SSIM']
    printstring = f"File,model,MSE,RMSE"
    for key in ssim_keys:
        printstring = f"{printstring},{key}"
    print(printstring)
    p = Pool(10)
    inp = []
    for file in z:
        inp.append((file, root, names, dataset_names))
    results = p.map(Pool_func_ms_SSIM, inp)
    for res in results:
        for r in res:
            print(r)


def Pool_func_ms_SSIM(inp):
    file, root, names, dataset_names = inp

    result = h5py.File(f"{root}/{file}.h5", 'r')
    Truth = np.array(result["truth"])
    lst = []
    for i, f in enumerate(names):

        if dataset_names[i] == "pix2pix":
            Prediction = np.array([result["pix2pix"][1], result["pix2pix"][0]]) / 255
        else:
            Prediction = np.array(result[dataset_names[i]])

        SSIM_normal = ms_ssim(torch.Tensor(np.array([Truth])), torch.Tensor(np.array([Prediction])), data_range=1,
                              size_average=False).item()

        SSIM_m = ms_ssim(torch.Tensor(np.array([Truth[0]])), torch.Tensor(np.array([Prediction[0]])), data_range=1,
                              size_average=False).item()
        SSIM_n = ms_ssim(torch.Tensor(np.array([Truth[1]])), torch.Tensor(np.array([Prediction[1]])), data_range=1,
                              size_average=False).item()
        MSE = mse(Truth, Prediction)
        RMSE = rmse(Truth, Prediction)

        lst.append(f"{file},{f},{MSE},{RMSE},{SSIM_normal},{SSIM_m},{SSIM_n}")
    return lst


def compare_models_masked_SSIM(root, name, dataset_name):
    z = [x.split("/") for x in
         glob(root + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    z = [x[0:len(x) - 3] for x in z]
    ssim_keys = ['SSIM_masked', 'Dice_masked', 'MSSIM_masked', 'MDice_masked', 'NSSIM_masked', 'NDice_masked', 'MSSIM',
                 'NSSIM', 'SSIM']
    printstring = f"File,model,MSE,RMSE"
    for key in ssim_keys:
        printstring = f"{printstring},{key}"
    print(printstring)

    for file in z:
        result = h5py.File(f"{root}/{file}.h5", 'r')
        if dataset_name == "pix2pix":
            Prediction = np.array([result["pix2pix"][1], result["pix2pix"][0]]) / 255
        else:
            Prediction = np.array(result[dataset_name])
        Truth = np.array(result["truth"])

        SSIM_normal = SSIM_result(Prediction, Truth)

        MSE = mse(Truth, Prediction)
        RMSE = rmse(Truth, Prediction)
        print(f"{file},{name},{MSE},{RMSE},{SSIM_normal}")


def compare_models_truth_mask(root, names, dataset_names):
    z = [x.split("/") for x in
         glob(root + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    z = [x[0:len(x) - 3] for x in z]
    ssim_keys = ['SSIM_masked', 'MSSIM_masked', 'NSSIM_masked', 'SSIM_blurred', 'MSSIM_blurred', 'NSSIM_blurred']
    printstring = f"File,model"
    for key in ssim_keys:
        printstring = f"{printstring},{key}"
    print(printstring)
    p = Pool(10)
    inp = []
    for file in z:
        inp.append((file, root, names, dataset_names))
    results = p.map(Pool_func, inp)
    for res in results:
        for r in res:
            print(r)


def Pool_func(inp):
    file, root, names, dataset_names = inp

    result = h5py.File(f"{root}/{file}.h5", 'r')
    Truth = np.array(result["truth"])
    Truth_mask_1 = sauvola_mask(Truth[1])
    Truth_mask_0 = sauvola_mask(Truth[0])
    Truth_mask = Truth_mask_1
    Truth_mask[Truth_mask_0] = True

    blurred_truth = gaussian_filter(Truth, sigma=1.25)
    lst = []
    for i, f in enumerate(names):

        if dataset_names[i] == "pix2pix":
            Prediction = np.array([result["pix2pix"][1], result["pix2pix"][0]]) / 255
        else:
            Prediction = np.array(result[dataset_names[i]])

        SSIM = Masked_SSIM_AVG_TruthMask(Prediction, Truth, Truth_mask)
        SSIM_1 = Masked_SSIM_TruthMask(Prediction[1], Truth[1], Truth_mask_1)
        SSIM_0 = Masked_SSIM_TruthMask(Prediction[0], Truth[0], Truth_mask_0)

        SSIM_blurred = Masked_SSIM_AVG_TruthMask(Prediction, blurred_truth, Truth_mask)
        SSIM_blurred_1 = Masked_SSIM_TruthMask(Prediction[1], blurred_truth[1], Truth_mask_1)
        SSIM_blurred_0 = Masked_SSIM_TruthMask(Prediction[0], blurred_truth[0], Truth_mask_0)

        lst.append(f"{file},{f},{SSIM},{SSIM_0},{SSIM_1},{SSIM_blurred},{SSIM_blurred_0},{SSIM_blurred_1}")
    return lst


def SSIM_result(Prediction, Truth):
    VMSSIM_masked, VMDice_masked, _, _ = ssim_scoring.Masked_SSIM(Prediction[0], Truth[0])
    VCSSIM_masked, VCDice_masked, _, _ = ssim_scoring.Masked_SSIM(Prediction[1], Truth[1])
    VMSSIM = ssim_scoring.ssim(Prediction[0], Truth[0], win_size=7, channel_axis=0)
    VCSSIM = ssim_scoring.ssim(Prediction[1], Truth[1], win_size=7, channel_axis=0)
    VSSIM = ssim_scoring.ssim(Prediction, Truth, win_size=7, channel_axis=0)

    VSSIM_masked, VDice_masked, _, _ = ssim_scoring.Masked_SSIM_AVG(Prediction, Truth)
    return f"{VSSIM_masked},{VDice_masked},{VMSSIM_masked},{VMDice_masked},{VCSSIM_masked},{VCDice_masked},{VMSSIM},{VCSSIM},{VSSIM}"


if __name__ == '__main__':
    # Commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="", help="Enter file path for .czi file")
    args = parser.parse_args()

    # Open the file and load the file
    data = []
    with open(args.c) as f:
        data = yaml.load(f, Loader=SafeLoader)

    if data["type"] == "masked_SSIM":
        compare_models_masked_SSIM(data["input"], data["name"], data["dataset_name"])
    if data["type"] == "masked_SSIM_truth_mask":
        compare_models_truth_mask(data["input"], data["name"], data["dataset_name"])
    if data["type"] == "ms_SSIM":
        compare_models_MS_SSIM(data["input"], data["name"], data["dataset_name"])
