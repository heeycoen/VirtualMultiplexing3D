
import h5py
import numpy as np

import argparse
import yaml
from yaml.loader import SafeLoader
from multiprocessing import Pool
from glob import glob


def h5_files(filedir):
    """
    list all h5 file names in the given directory
    :param filedir: the directory with h5 files
    :return:
    """
    z = [x.split("/") for x in
         glob(filedir + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    return [x[0:len(x) - 3] for x in z]



def copy_h5_dataset(output, input, input_dataset_name, new_dataset_name, pool_size=10):
    """
    copy the datasets from h5 files in the input directory to the output directory (the same file names need to be present)
    :param output: the output directory
    :param input: the input directory
    :param input_dataset_name: the name of the dataset in the input h5 file
    :param new_dataset_name: the name of the dataset in the output h5 file
    :param pool_size: default: 10, number of threads working at the same time
    """
    z = h5_files(input)
    poolinfo = []
    for file in z:
        poolinfo.append((output, file, input_dataset_name, new_dataset_name, input))
    p = Pool(pool_size)
    p.map(__pool_func_combine__, poolinfo)


def __pool_func_combine__(x):
    output, file, input_name, segment_name, inp = x

    outfile = h5py.File(f"{output}/{file}.h5", 'a')
    infile = h5py.File(f"{inp}/{file}.h5", 'r')
    dataset = np.array(infile[input_name])
    if not outfile.keys().__contains__(segment_name):
        outfile.create_dataset(name=segment_name, data=dataset)
    else:
        print(segment_name, "already exists")
    outfile.close()
    infile.close()



if __name__ == '__main__':
    # Commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="", help="Enter file path for .czi file")
    args = parser.parse_args()

    # Open the file and load the file
    data = []
    with open(args.c) as f:
        data = yaml.load(f, Loader=SafeLoader)

    if data["process"] == "combine":
        copy_h5_dataset(data["outputpath"], data["inputpath"], data["input_name"], data["output_name"])
