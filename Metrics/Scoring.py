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
def compare_models(root):
    z = [x.split("/") for x in
         glob(root + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    z = [x[0:len(x) - 3] for x in z]
    print(z)
    print("File,VMSE,PMSE,VRMSE,PRMSE")
    for file in z:
        result = h5py.File(f"{root}/{file}.h5", 'r')
        PixPrediction = np.array([result["pix2pix"][1], result["pix2pix"][0]]) / 255
        VoxPrediction = np.array(result["prediction"])
        Truth = np.array(result["truth"])

            # VMSSIM_masked, VMDice_masked, PMSSIM_masked, PMDice_masked = ssim_scoring.Masked_SSIM_SingleChan(VoxPrediction[0], PixPrediction[0], Truth[0])
            # VCSSIM_masked, VCDice_masked, PCSSIM_masked, PCDice_masked = ssim_scoring.Masked_SSIM_SingleChan(VoxPrediction[1], PixPrediction[1], Truth[1])
            # VMSSIM = ssim_scoring.ssim(VoxPrediction[0], Truth[0], win_size=7, channel_axis=0)
            # PMSSIM = ssim_scoring.ssim(PixPrediction[0], Truth[0], win_size=7, channel_axis=0)
            # VCSSIM = ssim_scoring.ssim(VoxPrediction[1], Truth[1], win_size=7, channel_axis=0)
            # PCSSIM = ssim_scoring.ssim(PixPrediction[1], Truth[1], win_size=7, channel_axis=0)
            # VSSIM = ssim_scoring.ssim(VoxPrediction, Truth, win_size=7, channel_axis=0)
            # PSSIM = ssim_scoring.ssim(PixPrediction, Truth, win_size=7, channel_axis=0)
            #
            # VSSIM_masked, VDice_masked,_,_ = ssim_scoring.Masked_SSIM_AVG(VoxPrediction, Truth)
            # PSSIM_masked, PDice_masked,_,_ = ssim_scoring.Masked_SSIM_AVG(PixPrediction, Truth)

        PMSE = mse(Truth, PixPrediction)
        VMSE = mse(Truth, VoxPrediction)
        PRMSE = rmse(Truth, PixPrediction)
        VRMSE = rmse(Truth, VoxPrediction)
        print(f"{file},{VMSE},{PMSE},{VRMSE},{PRMSE}")



if __name__ == '__main__':
    # Commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="", help="Enter file path for .czi file")
    args = parser.parse_args()

    # Open the file and load the file
    data = []
    with open(args.c) as f:
        data = yaml.load(f, Loader=SafeLoader)

    compare_models(data["input"])
