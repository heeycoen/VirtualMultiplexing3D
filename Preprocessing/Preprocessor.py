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


def normalize(patch, percentiles, c, normalization, n=1.0):
    """
    Normalize patch input, based on the percentiles given
    :param patch:
    :param percentiles: the percentiles for each channel
    :param c: channel to normalize
    :param normalization: normalization type, ac = use current channel percentile, eq = use highest percentile
    :param n: normalization max
    :return:
    """
    if normalization == "ac":
        # Normalizing the patch using the percentile of the current channel "c"
        normalized_patch = (patch / percentiles[c]) * n
    elif normalization == "eq":
        # Normalizing the patch using the highest percentile
        normalized_patch = (patch / max(percentiles)) * n
    else:
        return patch
    # Clipping the values, anything higher than 1 is set to 1
    normalized_patch[normalized_patch > n] = n
    return normalized_patch


def preprocess(input_path, output_path, img_width, img_height, img_depth_bottom, img_depth_top, calculate_percentile=False,
               percentile=99.8,
               channels=3, outputchannels=2,
               normalization="ac", norm_max=1.0):
    """
    Generates the mv training, testing and validation set
    :param input_path: the path to the czi file to convert
    :param output_path: where to return the results
    :param img_width: image width
    :param img_height: image height
    :param img_depth_bottom: bottom layer
    :param img_depth_top: top layer
    :param calculate_percentile: boolean whether to calculate percentile or use default normalization values (default False)
    :param percentile: the percentile to generate (default 99.8)
    :param channels: how many channels are present (default 3)
    :param outputchannels: how many unmixed channels (default 2)
    :param normalization: normalization type, ac = use current channel percentile, eq = use highest percentile (default ac)
    :param norm_max: normalization max (default normalize 0-1 float)
    """
    # make all h5py files, then fill in each plane of the h5py files.
    # by opening and closing the files it is less ram heavy
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    percentiles = []
    if calculate_percentile:

        print('*****Start Perc Calc*****')
        sys.stdout.write("\rExtracting the percentiles")
        for channel in range(channels):
            sys.stdout.write("\rChannel:[%d/%d]" % (channel, channels))
            tmp_channel_stacked_planes = []
            # For each layer in the range of lower and uper layer limits
            for idx in range(img_depth_bottom, img_depth_top):
                sys.stdout.write("\rLoading img: [%d/%d]"
                                 % (idx - img_depth_bottom + 1, img_depth_top - img_depth_bottom))
                # Extracting layer (or plane) using the reading function from stapl3D.
                data = shading.read_tiled_plane(input_path, channel, idx)

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
            planes.append(np.array(shading.read_tiled_plane(input_path, ch, idx)))
        sys.stdout.write("\rLoaded: [%d/%d]"
                         % (idx - img_depth_bottom + 1, img_depth_top - img_depth_bottom))
        for i in range(int(planes[0].shape[1] / img_width)):
            for j in range(int(planes[0].shape[2] / img_height)):
                for z in range(planes[0].shape[0]):
                    if os.path.exists(f'{output_path}/{z}_{i}-{j}.h5'):
                        hf = 'r+'
                    else:
                        hf = 'w'
                    with h5py.File(f'{output_path}/{z}_{i}-{j}.h5', hf) as df:
                        if channels > 1:
                            if not df.keys().__contains__('unmixed'):
                                if norm_max == 1.0:
                                    df.create_dataset('unmixed',
                                                      shape=[outputchannels, img_depth_top - img_depth_bottom,
                                                             img_width, img_height], dtype=float)
                                else:
                                    df.create_dataset('unmixed',
                                                      shape=[outputchannels, img_depth_top - img_depth_bottom,
                                                             img_width, img_height], dtype=int)

                            unm = df['unmixed']
                            for ch in range(outputchannels):
                                unm[ch, idx - img_depth_bottom, 0:img_width, 0:img_height] = normalize(
                                    (planes[ch][z, i * img_width: img_width + i * img_width,
                                     j * img_height: img_height + j * img_height]), percentiles, ch,
                                    normalization, norm_max)

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

                        m[0, idx - img_depth_bottom, 0:img_width, 0:img_height] = normalize(planes[channels - 1][z,
                                                                                            i * img_width: img_width + i * img_width,
                                                                                            j * img_height: img_height + j * img_height],
                                                                                            percentiles, channels - 1,
                                                                                            normalization, norm_max)
                        df.close()


def split_dataset(dataset_path, seed):
    """
    Split the dataset into train, test and validation
    :param dataset_path: path to dataset to split
    :param seed: shuffle seed
    """
    z = [x.split("/") for x in
         glob(dataset_path + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(dataset_path + "/train"):
        os.mkdir(dataset_path + "/train")
    if not os.path.exists(dataset_path + "/test"):
        os.mkdir(dataset_path + "/test")
    if not os.path.exists(dataset_path + "/validation"):
        os.mkdir(dataset_path + "/validation")
    os.chdir(dataset_path)

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


def filter_out_empty(dataset_path, threshold):
    """
    Filter out files that are below the threshold
    :param dataset_path: path to dataset to filter
    :param threshold: percentile threshold
    """
    z = list_h5_files(dataset_path)
    smlist = []
    smklist = []
    for x, file in enumerate(z):
        f = h5py.File(dataset_path + "/" + file, 'r')
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

    if os.path.exists(dataset_path + "/empty") == False:
        os.mkdir(dataset_path + "/empty")
    for fl in todel:
        os.rename(f"{dataset_path}/{fl}", f"{dataset_path}/empty/{fl}")
    smlist.sort()
    plt.plot(smlist)
    p.sort()
    plt.plot(p)
    plt.savefig(dataset_path + "/sum_fig.png")


def move_files(dataset_path, target_path):
    """
    Move the files in the dataset_path to the target_path
    :param dataset_path: the root dataset path
    :param target_path: the target path
    """
    z = [x.split("/") for x in
         glob(dataset_path + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    if os.path.exists(target_path) == False:
        os.mkdir(target_path)
    for fl in z:
        os.rename(f"{dataset_path}/{fl}", f"{target_path}/{fl}")


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
        thres = data["filter"]["threshold"]
    split = f"_NoEmpty_{thres}" if "filter" in data.keys() else ""
    outpch = data["outpChannels"]
    addition = data["SuffixResultname"]
    resultpath = f"{res}/{w}x{h}x{d}x{outpch}_P-{perc}_N{norm}-{n}{denoise}{dup}{split}_{addition}"

    if data["generate"]:
        preprocess(data["Filepath"], resultpath, data["img_width"], data["img_height"],
                   data["img_depth_bottom"],
                   data["img_depth_top"], data["calc_perc"], data["percentile"], data["channels"],
                   data["outpChannels"],
                   data["normalization"], data["norm_max"])
    if "filter" in data.keys():
        filter_out_empty(resultpath, data["filter"]["threshold"])
    if "splitgen" in data.keys():
        split_dataset(resultpath, data["splitgen"]["random_seed"])
    if "movefiles" in data.keys():
        move_files(data["movefiles"]["root"], data["movefiles"]["target"])
