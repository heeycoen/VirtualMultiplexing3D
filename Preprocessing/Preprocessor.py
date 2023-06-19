import numpy as np
from stapl3d.preprocessing import shading
import h5py
import argparse
import cv2
import os
from sklearn.model_selection import train_test_split
from glob import glob
import sys
import yaml
from yaml.loader import SafeLoader
from matplotlib import pyplot as plt


def normalize(patch, percentiles, c, normalization, n = 1.0):
    if normalization == "ac":
        # Normalizing the patch using the percentile of the current channel "c"
        normalized_patch = (patch / percentiles[c]) * n
    elif normalization == "eq":
        # Normalizing the patch using the highest percentile (open detector channel percentile)
        normalized_patch = (patch / percentiles[2]) * n
    else:
        return patch
    # Clipping the values, anything higher than 1 is set to 1
    normalized_patch[normalized_patch > n] = n
    return normalized_patch


def denoise_in(input, denoise):
    if denoise:
        return cv2.fastNlMeansDenoising(input.astype('uint8'), None, 10, 7, 21)
    else:
        return input

def synthetic_data(ch1, ch2, method):
    ch3 = np.empty_like(ch1)
    if method == "50-50":
        ch3 = ch1+ch2/2
    elif method == "Max":
        ch3 = np.maximum(ch1, ch2)
    return ch3

def main_Datagen(filepath, resultpath, img_width, img_height, img_depth_bottom, img_depth_top, calcperc, percentile,
                 channels, outputchannels,
                 normalization, norm_max, denoise, dupe):
    # make all h5py files, then fill in each plane of the h5py files.
    # by opening and closing the files it is less ram heavy
    if os.path.exists(resultpath) == False:
        os.mkdir(resultpath)
    percentiles = []
    if calcperc:

        print('*****Start Perc Calc*****')
        sys.stdout.write("\rExtracting the percentiles")
        for channel in range(channels):
            sys.stdout.write("\rChannel:[%d/%d]"%(channel, channels))
            tmp_channel_stacked_planes = []
            # For each layer in the range of lower and uper layer limits
            for idx in range(img_depth_bottom, img_depth_top):
                sys.stdout.write("\rLoading img: [%d/%d]"
                         % (idx - img_depth_bottom + 1, img_depth_top - img_depth_bottom))
                # Extracting layer (or plane) using the reading function from stapl3D. The plane consist of 49 tiles.
                data = shading.read_tiled_plane(filepath, channel, idx)
                # Stacking the 49 tiles on top of each other.
                dstacked = np.stack(data, axis=0)
                # Add stacked tiles from the single plane to temp list collecting each plane
                tmp_channel_stacked_planes.append(dstacked)

                # Stacking all collected planes from a single channel as the following dimensions l,y,x
                planes_stacked = np.vstack(tmp_channel_stacked_planes)
            # Calculating percentile value for the channel "i"
            p_val = np.percentile(planes_stacked, percentile)

            # Append percentile values to list
            percentiles.append(p_val)
            sys.stdout.write("\nChannel:[%d/%d] Percentile: [%d]" % (channel, channels, p_val))
        print('*****Done Perc Calc*****')
    else:

        percentiles.append(15692)
        percentiles.append(9303)
        percentiles.append(18980)

    for idx in range(img_depth_bottom, img_depth_top):
        sys.stdout.write("\nLoading img: [%d/%d]"
                         % (idx - img_depth_bottom + 1, img_depth_top - img_depth_bottom))
        planes = []
        for ch in range(channels):
            planes.append(np.array(shading.read_tiled_plane(filepath, ch, idx)))
        sys.stdout.write("\rLoaded: [%d/%d]"
                         % (idx - img_depth_bottom + 1, img_depth_top - img_depth_bottom))
        for i in range(int(planes[0].shape[1] / img_width)):
            for j in range(int(planes[0].shape[2] / img_height)):
                for z in range(planes[0].shape[0]):
                    if os.path.exists(f'{resultpath}/{z}_{i}-{j}.h5'):
                        hf = 'r+'
                    else:
                        hf = 'w'
                    with h5py.File(f'{resultpath}/{z}_{i}-{j}.h5', hf) as df:
                        if channels > 1:
                            if not df.keys().__contains__('unmixed'):
                                if norm_max == 1.0:
                                    df.create_dataset('unmixed',
                                              shape=[outputchannels, img_depth_top - img_depth_bottom, img_width, img_height], dtype=float)
                                else:  df.create_dataset('unmixed',
                                              shape=[outputchannels, img_depth_top - img_depth_bottom, img_width, img_height], dtype=int)


                            unm = df['unmixed']
                            for ch in range(outputchannels):
                                unm[ch, idx - img_depth_bottom, 0:img_width, 0:img_height] = denoise_in(
                                    normalize((planes[ch][z, i * img_width: img_width + i * img_width,
                                               j * img_height: img_height + j * img_height]), percentiles, ch,
                                              normalization,norm_max)
                                    , denoise)

                        if not df.keys().__contains__('mixed'):
                            if norm_max == 1.0:
                                df.create_dataset('mixed',
                                                  shape=[outputchannels, img_depth_top - img_depth_bottom, img_width,
                                                         img_height], dtype=float)
                            else:
                                df.create_dataset('mixed',
                                                  shape=[outputchannels, img_depth_top - img_depth_bottom, img_width,
                                                         img_height], dtype=int)

                        m = df['mixed']

                        m[0, idx - img_depth_bottom, 0:img_width, 0:img_height] = denoise_in(
                            normalize(planes[channels-1][z,
                                      i * img_width: img_width + i * img_width,
                                      j * img_height: img_height + j * img_height], percentiles, channels-1, normalization, norm_max),
                            denoise)
                        if dupe and outputchannels > 1:
                            for ch in range(1,outputchannels):
                                m[ch, idx - img_depth_bottom, 0:img_width, 0:img_height] = m[0, idx - img_depth_bottom, 0:img_width, 0:img_height]
                        df.close()


def split_dataset(resultpath, seed):
    z = [x.split("/") for x in
         glob(resultpath + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    datafoldername = resultpath
    if os.path.exists(datafoldername) == False:
        os.mkdir(datafoldername)
    if os.path.exists(datafoldername + "/train") == False:
        os.mkdir(datafoldername + "/train")
    if os.path.exists(datafoldername + "/test") == False:
        os.mkdir(datafoldername + "/test")
    if os.path.exists(datafoldername + "/validation") == False:
        os.mkdir(datafoldername + "/validation")
    os.chdir(datafoldername)

    train, test = train_test_split(z, test_size=0.4, shuffle=True, random_state=seed)
    test, validate = train_test_split(test, test_size=0.5, shuffle=True, random_state=seed)

    for file in train:
        os.rename(f"{file}", f"train/{file}")

    for file in test:
        os.rename(f"{file}", f"test/{file}")

    for file in validate:
        os.rename(f"{file}", f"validation/{file}")

def list_h5_files(resultpath):
    z = [x.split("/") for x in
         glob(resultpath + "/*.h5", recursive=True)]
    return [x[len(x) - 1] for x in z]

def filter(resultpath, threshold):
    z = list_h5_files(resultpath)
    smlist = []
    smklist = []
    for x, file in enumerate(z):
        f = h5py.File(resultpath + "/" + file, 'r')
        unmixedSm = np.sum(f['unmixed'])
        mixedSm = np.sum(f['mixed'])
        smlist.append(unmixedSm + mixedSm)
        smklist.append(file)
        sys.stdout.write("\rSum file %s, [mixed: %d] [unmixed: %d]"
                         % (file, mixedSm, unmixedSm))
    perc = np.percentile(smlist, threshold)

    ar = np.array(smlist)
    kar = np.array(smklist)
    p = ar[ar > perc]
    todel = kar[ar <= perc]

    if os.path.exists(resultpath + "/empty") == False:
        os.mkdir(resultpath + "/empty")
    for fl in todel:
        os.rename(f"{resultpath}/{fl}", f"{resultpath}/empty/{fl}")
    smlist.sort()
    plt.plot(smlist)
    p.sort()
    plt.plot(p)
    plt.savefig(resultpath + "/sum_fig.png")

def moveFiles(root, target):
    z = [x.split("/") for x in
         glob(root + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    if os.path.exists(target) == False:
        os.mkdir(target)
    for fl in z:
        os.rename(f"{root}/{fl}", f"{target}/{fl}")


if __name__ == '__main__':
    # Commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="", help="Enter file path for .czi file")
    args = parser.parse_args()

    # Open the file and load the file
    data = []
    with open(args.c) as f:
        data = yaml.load(f, Loader=SafeLoader)

    w = data["img_width"]
    h = data["img_height"]
    d = data["img_depth_top"] - data["img_depth_bottom"]
    perc = data["percentile"]
    norm = data["normalization"]
    n = data["norm_max"]
    denoise = "_Denoised" if data["denoise"] else ""
    res = data["Resultpath"]
    dup = "_Dupe1stChan" if data["duplicated"] else ""
    if "filter" in data.keys():
        thres= data["filter"]["threshold"]
    split = f"_NoEmpty_{thres}" if "filter" in data.keys() else ""
    outpch = data["outpChannels"]
    addition = data["SuffixResultname"]
    resultpath = f"{res}/{w}x{h}x{d}x{outpch}_P-{perc}_N{norm}-{n}{denoise}{dup}{split}_{addition}"

    if data["generate"]:
        main_Datagen(data["Filepath"], resultpath, data["img_width"], data["img_height"], data["img_depth_bottom"],
                     data["img_depth_top"], data["calc_perc"], data["percentile"], data["channels"], data["outpChannels"],
                     data["normalization"], data["norm_max"], data["denoise"], data["duplicated"])
    if "filter" in data.keys():
        filter(resultpath, data["filter"]["threshold"])
    if "splitgen" in data.keys():
        split_dataset(resultpath, data["splitgen"]["random_seed"])
    if "movefiles" in data.keys():
        moveFiles(data["movefiles"]["root"], data["movefiles"]["target"])

