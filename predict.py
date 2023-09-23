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


def predict(dl, generator, outpath):
    """
    predict the files in the dataloader into files with the image and prediction
    :param dl:
    :param generator:
    :param outpath:
    """
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for i, batch in enumerate(dl):
        input_img = Variable(batch["A"].type(Tensor))

        # generator forward pass
        generated_image = generator(input_img)
        fake_B = generated_image.cpu().detach().numpy()[0]
        real_A = input_img.cpu().detach().numpy()[0]

        image_folder = "%s/%d_" % (outpath, i)

        hf = h5py.File(image_folder + 'result.h5', 'w')
        hf.create_dataset('image', data=real_A)
        hf.create_dataset('prediction', data=fake_B)
        hf.close()


def predict_full(dl, generator, output_path):
    """
    Predict the files in the dataloader into files with both the images, truth and prediction
    :param dl: dataloader
    :param generator: pytorch generator
    :param output_path:
    """

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for i, batch in enumerate(dl):
        input_img = Variable(batch["A"].type(Tensor))

        target_img = Variable(batch["B"].type(Tensor))
        # generator forward pass
        generated_image = generator(input_img)
        fake_B = generated_image.cpu().detach().numpy()[0]
        real_B = target_img.cpu().detach().numpy()[0]
        real_A = input_img.cpu().detach().numpy()[0]

        image_folder = "%s/%d_" % (output_path, i)

        hf = h5py.File(image_folder + 'result.h5', 'w')
        hf.create_dataset('image', data=real_A)
        hf.create_dataset('truth', data=real_B)
        hf.create_dataset('prediction', data=fake_B)
        hf.close()


def load_model(filepath, channels):
    generator = GeneratorUNet(channels, channels)
    generator.load_state_dict(
        torch.load(filepath))
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator = generator.cuda()

    return generator


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

    generator = load_model(data["model_path"], data["channels"])

    if os.path.exists(data["output_path"]) == False:
        os.mkdir(data["output_path"])

    if data["predict"]:
        dl = DataLoader(
            PredictDataset(data["dataset_path"]),
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

        predict(dl, generator, data["output_path"])

    if data["predict_full"]:
        dl = DataLoader(
            CTDataset(data["dataset_path"]),
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

        predict_full(dl, generator, data["output_path"])
