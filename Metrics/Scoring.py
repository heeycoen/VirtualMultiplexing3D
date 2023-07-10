import ssim_scoring
import os
from glob import glob
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error
import yaml
from yaml.loader import SafeLoader
import argparse
def compare_models(root):
    z = [x.split("/") for x in
         glob(root + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    z = [x[0:len(x) - 3] for x in z]

    print("File, VMSE, PMSE, VSSIM_masked,PSSIM_masked,VMSSIM_masked,VMDice_masked,PMSSIM_masked,PMDice_masked,VCSSIM_masked,VCDice_masked,PCSSIM_masked,PCDice_masked,VMSSIM,PMSSIM,VCSSIM,PCSSIM,VSSIM,PSSIM")

    for file in z:
        result = h5py.File(f"{root}/{file}", 'r')
        PixPrediction = np.array([result["pix2pix"][1], result["pix2pix"][0]]) / 255
        VoxPrediction = np.array(result["prediction"])
        Truth = np.array(result["truth"])

        VMSSIM_masked, VMDice_masked, PMSSIM_masked, PMDice_masked = ssim_scoring.Masked_SSIM_SingleChan(VoxPrediction[0], PixPrediction[0], Truth[0])
        VCSSIM_masked, VCDice_masked, PCSSIM_masked, PCDice_masked = ssim_scoring.Masked_SSIM_SingleChan(VoxPrediction[1], PixPrediction[1], Truth[1])
        VMSSIM = ssim_scoring.ssim(VoxPrediction[0], Truth[0], win_size=7, channel_axis=0)
        PMSSIM = ssim_scoring.ssim(PixPrediction[0], Truth[0], win_size=7, channel_axis=0)
        VCSSIM = ssim_scoring.ssim(VoxPrediction[1], Truth[1], win_size=7, channel_axis=0)
        PCSSIM = ssim_scoring.ssim(PixPrediction[1], Truth[1], win_size=7, channel_axis=0)
        VSSIM = ssim_scoring.ssim(VoxPrediction, Truth, win_size=7, channel_axis=0)
        PSSIM = ssim_scoring.ssim(PixPrediction, Truth, win_size=7, channel_axis=0)

        VSSIM_masked = ssim_scoring.Masked_SSIM_AVG(VoxPrediction, Truth)
        PSSIM_masked = ssim_scoring.Masked_SSIM_AVG(PixPrediction, Truth)

        PMSE = mean_squared_error(Truth, PixPrediction)
        VMSE = mean_squared_error(Truth, VoxPrediction)
        print(f"{file},{VMSE},{PMSE},{VSSIM_masked},{PSSIM_masked},{VMSSIM_masked},{VMDice_masked},{PMSSIM_masked},{PMDice_masked},{VCSSIM_masked},{VCDice_masked},{PCSSIM_masked},{PCDice_masked},{VMSSIM},{PMSSIM},{VCSSIM},{PCSSIM},{VSSIM},{PSSIM}")




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
