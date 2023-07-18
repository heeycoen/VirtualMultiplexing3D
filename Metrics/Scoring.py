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
def compare_models(root,name,dataset_name):
    z = [x.split("/") for x in
         glob(root + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    z = [x[0:len(x) - 3] for x in z]
    ssim_keys = ['SSIM_masked','Dice_masked','MSSIM_masked','MDice_masked','NSSIM_masked','NDice_masked','MSSIM','NSSIM','SSIM']
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

        SSIM_normal = SSIM_result(Prediction,Truth)

        MSE = mse(Truth, Prediction)
        RMSE = rmse(Truth, Prediction)
        print(f"{file},{name},{MSE},{RMSE},{SSIM_normal}")

def SSIM_result(Prediction, Truth):
    VMSSIM_masked, VMDice_masked ,_,_= ssim_scoring.Masked_SSIM(Prediction[0], Truth[0])
    VCSSIM_masked, VCDice_masked,_,_= ssim_scoring.Masked_SSIM(Prediction[1], Truth[1])
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

    compare_models(data["input"],data["name"],data["dataset_name"])
