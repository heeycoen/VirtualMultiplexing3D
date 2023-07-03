import argparse
from glob import glob

import matplotlib
import os
import numpy as np
import h5py
import yaml
from PIL import Image
from numpy import asarray
from yaml.loader import SafeLoader


def convert_to_pix2pix(file, data):

    datafolder = data + "/pix2pix_conversion"
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)

    test = h5py.File(data + "/" + file + ".h5", 'r')

    x = test["image"][0]
    for i,layer in enumerate(x):
        source_image = np.zeros((layer.shape[0], layer.shape[1], 3))
        layer *= 255
        source_image[:, :, 0] = layer
        source_image[:, :, 1] = layer
        source_image[:, :, 2] = layer
        matplotlib.image.imsave(datafolder + f"/{file}_layer_" + str(i + 1) + '.png', source_image.astype(np.uint8))

def batch_convert_to_pix2pix(root):
    z = [x.split("/") for x in
         glob(root + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    z = [x[0:len(x) - 3] for x in z]
    for file in z:
        convert_to_pix2pix(file, root)

def batch_convert_from_pix2pix(root):
    z = [x.split("/") for x in
         glob(root + "/*.h5", recursive=True)]
    z = [x[len(x) - 1] for x in z]
    z = [x[0:len(x) - 3] for x in z]
    for file in z:
        convert_from_pix2pix_results(root,file,128)



def convert_from_pix2pix_results(root, filename, nr):
    z = [filename+"_layer_%s_fake.png" % (x+1) for x in range(nr)]

    with h5py.File(f'{root}/{filename}.h5', 'r+') as df:
        df.create_dataset('pix2pix',
                          shape=[3, 128, 512, 512], dtype=float)
        mx = df["pix2pix"]
        i = 0
        for fi in z:
            with Image.open(root + "/pix2pix_results/" + fi) as img:
                arr = asarray(img)
                mx[0, i, 0:512, 0:512] = arr[0:512, 0:512, 0]
                mx[1, i, 0:512, 0:512] = arr[0:512, 0:512, 1]
                mx[2, i, 0:512, 0:512] = arr[0:512, 0:512, 2]
            i += 1
        df.close()



