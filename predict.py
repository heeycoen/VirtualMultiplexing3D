import argparse
import os

import yaml
from yaml.loader import SafeLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from Model.Model import GeneratorUNet
from Model.dataset import CTDataset, PredictDataset

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import torch

import h5py

def test_generator(dl, generator, tensorboard_writer, Tensor,outpath):
    avgMSE = 0
    avgSSIM_M = 0
    avgSSIM_N = 0
    for i, batch in enumerate(dl):
        input_img = Variable(batch["A"].type(Tensor))
        target_img = Variable(batch["B"].type(Tensor))
        # generator forward pass
        generated_image = generator(input_img)
        fake_B = generated_image.cpu().detach().numpy()[0]
        real_B = target_img.cpu().detach().numpy()[0]
        mse = mean_squared_error(fake_B, real_B)
        ss_n = ssim(fake_B[1], real_B[1], winsize=(7, 7), channel_axis=0)
        ss_m = ssim(fake_B[0], real_B[0], winsize=(7, 7), channel_axis=0)
        avgMSE += mse
        avgSSIM_M += ss_m
        avgSSIM_N += ss_n
        real_A = input_img.cpu().detach().numpy()[0]

        image_folder = "%s/images/%d_" % (outpath, i)

        hf = h5py.File(image_folder + 'real_A.vox', 'w')
        hf.create_dataset('data', data=real_A)

        hf1 = h5py.File(image_folder + 'real_B.vox', 'w')
        hf1.create_dataset('data', data=real_B)

        hf2 = h5py.File(image_folder + 'fake_B.vox', 'w')
        hf2.create_dataset('data', data=fake_B)


    tensorboard_writer.add_scalars(
        "Gen_MSE",
        {"train": avgMSE / len(dl)},
    )
    tensorboard_writer.add_scalars(
        "Gen_SSIM",
        {"train_mem": avgSSIM_M / len(dl)},
    )
    tensorboard_writer.add_scalars(
        "Gen_SSIM",
        {"train_nuc": avgSSIM_N / len(dl)},
    )


def predict(dl, generator, outpath, Tensor):
    for i, batch in enumerate(dl):
        input_img = Variable(batch["A"].type(Tensor))
        # generator forward pass
        generated_image = generator(input_img)
        fake_B = generated_image.cpu().detach().numpy()[0]
        real_A = input_img.cpu().detach().numpy()[0]

        image_folder = "%s/%d_" % (outpath, i)

        hf = h5py.File(image_folder + 'real_A.vox', 'w')
        hf.create_dataset('data', data=real_A)

        hf2 = h5py.File(image_folder + 'fake_B.vox', 'w')
        hf2.create_dataset('data', data=fake_B)




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

    generator, Tensor = load_model(data["model_path"],data["channels"])

    if os.path.exists(data["output_path"]) == False:
        os.mkdir(data["output_path"])

    if data["predict"]:
        dl = DataLoader(
            PredictDataset(data["dataset_path"]),
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

        predict(dl,generator,data["output_path"], Tensor)

    elif data["testing"]:
        dl = DataLoader(
            CTDataset(data["dataset_path"]),
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

        tensorboard_writer = SummaryWriter(log_dir="%s/tb" % (data["output_path"]))
        test_generator(dl, generator, tensorboard_writer, Tensor, data["output_path"])