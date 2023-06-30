from glob import glob

import matplotlib
import os
import numpy as np
import h5py
from PIL import Image
from numpy import asarray


def convert_to_pix2pix(file, data):

    datafolder = data + "/pix2pix_conversion"
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)

    test = h5py.File(data + "/" + file + ".h5", 'r')

    y = np.array(test["unmixed"])
    for layer in range(y.shape[1]):
        source_image = np.zeros((y[0].shape[1], y[0].shape[2], 3))
        source_image[:, :, 0] = y[0, layer] * 255
        source_image[:, :, 1] = y[1, layer] * 255
        matplotlib.image.imsave(datafolder + f"/{file}_layer_" + str(layer + 1) + '.png', source_image.astype(np.uint8))

def convert_from_pix2pix_results(root, filename, nr):
    z = [filename+"_layer_%s_fake.png" % x for x in range(nr)]

    with h5py.File(f'{root}/resultImage.h5', 'w') as df:
        df.create_dataset('unmixed',
                          shape=[3, 128, 512, 512], dtype=float)
        mx = df["unmixed"]
        i = 0
        for fi in z:
            with Image.open(root + "/" + fi) as img:
                arr = asarray(img)
                mx[0, i, 0:512, 0:512] = arr[0:512, 0:512, 0]
                mx[1, i, 0:512, 0:512] = arr[0:512, 0:512, 1]
                mx[2, i, 0:512, 0:512] = arr[0:512, 0:512, 2]

            i += 1
        df.close()

