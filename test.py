import argparse
import os

import numpy as np
import yaml
from yaml.loader import SafeLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable

from Model.Model import GeneratorUNet
from Model.dataset import PredictDataset
from Metrics.ssim_scoring import Masked_SSIM_AVG, Masked_SSIM
import torch

import h5py


def predict(dl, generator, outpath, Tensor, modelnr):
    for i, batch in enumerate(dl):
        input_img = Variable(batch["A"].type(Tensor))
        # generator forward pass
        generated_image = generator(input_img)
        fake_B = generated_image.cpu().detach().numpy()[0]
        real_A = input_img.cpu().detach().numpy()[0]

        image_folder = "%s/%d_%d_" % (outpath, modelnr, i)

        hf = h5py.File(image_folder + 'real_A.vox', 'w')
        hf.create_dataset('data', data=real_A)

        hf2 = h5py.File(image_folder + 'fake_B.vox', 'w')
        hf2.create_dataset('data', data=fake_B)


def test(dl, generator, Tensor):
    VoxSSIMAVG = []
    VoxDiceAVG = []
    VNSSIMAVG = []
    VNDiceAVG = []
    VMSSIMAVG = []
    VMDiceAVG = []
    for i, batch in enumerate(dl):
        input_img = Variable(batch["A"].type(Tensor))
        # generator forward pass
        generated_image = generator(input_img)
        Prediction = generated_image.cpu().detach().numpy()[0]
        Real = Variable(batch["B"].type(Tensor)).cpu().detach().numpy()[0]

        VoxSSIM, VoxDice, _, _ = Masked_SSIM_AVG(Prediction, Real)
        VMSSIM, VMDice, _, _ = Masked_SSIM(Prediction[0], Real[0])
        VNSSIM, VNDice, _, _ = Masked_SSIM(Prediction[1], Real[1])
        VoxSSIMAVG.append(VoxSSIM)
        VoxDiceAVG.append(VoxDice)
        VNSSIMAVG.append(VNSSIM)
        VNDiceAVG.append(VNDice)
        VMSSIMAVG.append(VMSSIM)
        VMDiceAVG.append(VMDiceAVG)

    print(np.array(VoxSSIMAVG).mean())
    print(np.array(VoxDiceAVG).mean())
    print(np.array(VNSSIMAVG).mean())
    print(np.array(VNDiceAVG).mean())
    print(np.array(VMSSIMAVG).mean())
    print(np.array(VMDiceAVG).mean())


def load_model(filepath, channels):
    generator = GeneratorUNet(channels, channels)
    generator.load_state_dict(
        torch.load(filepath))
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator = generator.cuda()

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    return generator, Tensor


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="", help="Enter path for config file")

    args = parser.parse_args()
    # Open the file and load the file
    data = []
    with open(args.c) as f:
        data = yaml.load(f, Loader=SafeLoader)

    dl = DataLoader(
        PredictDataset(data["dataset_path"]),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    for models in data["model_paths"]:
        generator, Tensor = load_model(models, 2)
        test(dl, generator, Tensor)
