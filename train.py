import argparse
import os

import numpy as np
import time
import datetime
import sys

import torchvision.transforms as transforms
import yaml
from yaml.loader import SafeLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from Model.Model import GeneratorUNet, weights_init_normal, Discriminator
from Model.dataset import CTDataset

from Model.dice_loss import diceloss
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import torch

import h5py

adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()


def generator_loss(generated_image, target_img, G, real_target, lambda_g):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (lambda_g * l1_l)
    # print(gen_loss)
    return gen_total_loss


def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss


def train(epoch, n_epochs, pathname, output_path, dataset_name, batch_size, glr, dlr,
          b1, b2, n_cpu, img_height, img_width,
          img_depth, channels, sample_interval,
          checkpoint_interval, lambda_voxel, tensorboard_writer):
    """

    :param epoch: Start at epoch nr
    :param n_epochs: Number of epochs
    :param pathname: root path name
    :param output_path:
    :param dataset_name: name of dataset
    :param batch_size: batch size
    :param glr: generator learning rate
    :param dlr: discriminator learning rate
    :param b1:
    :param b2:
    :param n_cpu: number of cpu cores available
    :param img_height: image height
    :param img_width: image width
    :param img_depth: image depth
    :param channels: channels to train on
    :param sample_interval:
    :param checkpoint_interval: after how many epochs is a checkpoint created
    :param lambda_voxel:
    :param tensorboard_writer:
    """
    cuda = True if torch.cuda.is_available() else False

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, img_depth // 2 ** 4, img_height // 2 ** 4, img_width // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet(channels, channels)
    discriminator = Discriminator(channels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        adversarial_loss.cuda()
        l1_loss.cuda()
        sys.stdout.write(
            "\rCuda Enabled [Device %s]"
            % (
                device,
            )
        )
    else:
        sys.stdout.write(
            "\rCuda Disabled [Device %s]"
            % (
                device,
            )
        )
    if epoch != 0:
        # Load pretrained models
        generator.load_state_dict(
            torch.load("%s%s/saved_models/generator_%d.pth" % (output_path, dataset_name, epoch)))
        discriminator.load_state_dict(
            torch.load("%s%s/saved_models/discriminator_%d.pth" % (output_path, dataset_name, epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=glr, betas=(b1, b2))
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=dlr, betas=(b1, b2))
    sys.stdout.write(
        "\rG_optimizer: [lr %f] [Betas %f/%f]"
        % (
            glr,
            b1,
            b2
        )
    )
    sys.stdout.write(
        "\rD_optimizer: [lr %f] [Betas %f/%f]"
        % (
            dlr,
            b1,
            b2
        )
    )
    train_dl = DataLoader(
        CTDataset(pathname + "/%s/train/" % dataset_name),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    val_dl = DataLoader(
        CTDataset(pathname + "/%s/test/" % dataset_name),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    print('*****Start Training*****')
    prev_time = time.time()
    for epoch in range(1, n_epochs + 1):
        avgDLoss = 0
        avgGLoss = 0
        avgMSE = 0
        avgSSIM_M = 0
        avgSSIM_N = 0
        for i, batch in enumerate(train_dl):
            D_optimizer.zero_grad()
            input_img = Variable(batch["A"].type(Tensor))
            target_img = Variable(batch["B"].type(Tensor))

            # ground truth labels real and fake
            real_target = Variable(torch.ones(input_img.size(0), *patch).type(Tensor))
            fake_target = Variable(torch.zeros(input_img.size(0), *patch).type(Tensor))

            # generator forward pass
            generated_image = generator(input_img)
            g = generated_image.cpu().detach().numpy()
            t = target_img.cpu().detach().numpy()
            mse = mean_squared_error(g, t)
            ss_n = ssim(g[0][1], t[0][1], winsize=(7, 7), channel_axis=0)
            ss_m = ssim(g[0][0], t[0][0], winsize=(7, 7), channel_axis=0)
            avgMSE += mse
            avgSSIM_M += ss_m
            avgSSIM_N += ss_n
            # train discriminator with fake/generated images
            D_fake = discriminator(input_img, generated_image)

            D_fake_loss = discriminator_loss(D_fake.detach(), fake_target)

            # train discriminator with real images
            D_real = discriminator(input_img, target_img)
            D_real_loss = discriminator_loss(D_real, real_target)

            # average discriminator loss
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            # compute gradients and run optimizer step
            D_total_loss.backward()
            D_optimizer.step()

            # Train generator with real labels
            G_optimizer.zero_grad()
            G = discriminator(input_img, generated_image)
            G_loss = generator_loss(generated_image, target_img, G, real_target, lambda_voxel)
            # compute gradients and run optimizer step
            G_loss.backward()
            G_optimizer.step()

            batches_done = epoch * len(train_dl) + i

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_left = n_epochs * len(train_dl) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            avgGLoss += G_loss.item()
            avgDLoss += D_total_loss
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, MSE: %f, SSIM_M: %f, SSIM_N: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_dl),
                    D_total_loss,
                    G_loss.item(),
                    mse,
                    ss_m,
                    ss_n,
                    time_left,
                )
            )
        if sample_interval != -1 and epoch % sample_interval == 0:
            sample_voxel_volumes(epoch, val_dl, Tensor, generator, output_path, dataset_name)
        tensorboard_writer.add_scalars(
            "Gen_Loss",
            {"train": avgGLoss / len(train_dl)},
            epoch,
        )
        tensorboard_writer.add_scalars(
            "Gen_MSE",
            {"train": avgMSE / len(train_dl)},
            epoch,
        )
        tensorboard_writer.add_scalars(
            "Gen_SSIM",
            {"train_mem": avgSSIM_M / len(train_dl)},
            epoch,
        )

        tensorboard_writer.add_scalars(
            "Gen_SSIM",
            {"train_nuc": avgSSIM_N / len(train_dl)},
            epoch,
        )
        tensorboard_writer.add_scalars(
            "Dis_Loss",
            {"train": avgDLoss / len(train_dl)},
            epoch,
        )
        sys.stdout.write(
            "\r[Epoch %d/%d] [Average Losses] [D loss: %f] [G loss: %f, MSE: %f, SSIM_M: %f, SSIM_N: %f]"
            % (
                epoch,
                n_epochs,
                avgDLoss / len(train_dl),
                avgGLoss / len(train_dl),
                avgMSE / len(train_dl),
                avgSSIM_M / len(train_dl),
                avgSSIM_N / len(train_dl),
            )
        )
        print('*****volumes sampled*****')

        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(),
                       "%s%s/saved_models/generator_%d.pth" % (output_path, dataset_name, epoch))
            torch.save(discriminator.state_dict(),
                       "%s%s/saved_models/discriminator_%d.pth" % (output_path, dataset_name, epoch))


def sample_voxel_volumes(epoch, val_dl, Tensor, generator, output_path, dataset_name):
    """Saves a generated sample from the validation set"""
    avgSSIM = 0
    avgMSE = 0
    for i in range(10):
        imgs = next(iter(val_dl))
        real_A = Variable(imgs["A"].type(Tensor))
        real_B = Variable(imgs["B"].type(Tensor))
        fake_B = generator(real_A)

        # convert to numpy arrays
        real_A = real_A.cpu().detach().numpy()[0]
        real_B = real_B.cpu().detach().numpy()[0]
        fake_B = fake_B.cpu().detach().numpy()[0]

        mse = mean_squared_error(real_B, fake_B)
        ss = ssim(real_B, fake_B, winsize=(7, 7), channel_axis=0)
        avgMSE += mse
        avgSSIM += ss

    tensorboard_writer.add_image(
        "Val_real_A",
        real_A[0:2, int(real_A.shape[1] / 2)],
        global_step=epoch
    )
    tensorboard_writer.add_image(
        "Val_real_B",
        real_B[0:2, int(real_B.shape[1] / 2)],
        global_step=epoch
    )
    tensorboard_writer.add_image(
        "Val_fake_B",
        fake_B[0:2, int(fake_B.shape[1] / 2)],
        global_step=epoch
    )

    tensorboard_writer.add_scalars(
        "Gen_SSIM",
        {"test": avgSSIM / 10},
        epoch,
    )

    tensorboard_writer.add_scalars(
        "Gen_MSE",
        {"test": avgMSE / 10},
        epoch,
    )

    image_folder = "%s%s/images/epoch_%s_" % (output_path, dataset_name, epoch)

    hf = h5py.File(image_folder + 'real_A.vox', 'w')
    hf.create_dataset('data', data=real_A)

    hf1 = h5py.File(image_folder + 'real_B.vox', 'w')
    hf1.create_dataset('data', data=real_B)

    hf2 = h5py.File(image_folder + 'fake_B.vox', 'w')
    hf2.create_dataset('data', data=fake_B)


def create_path_names(output_path, dataset_name):
    os.makedirs("%s%s/images/" % (output_path, dataset_name), exist_ok=True)
    os.makedirs("%s%s/saved_models" % (output_path, dataset_name), exist_ok=True)
    os.makedirs("%s%s/tb" % (output_path, dataset_name), exist_ok=True)


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

    output = "%s/%s" % (data["output_path"], data["suffix"])

    create_path_names(output, data["dataset_name"])
    tensorboard_writer = SummaryWriter(log_dir="%s%s/tb" % (output, data["dataset_name"]))

    train(data["epoch"], data["n_epoch"], data["data_path"], output, data["dataset_name"],
          data["batch_size"], data["glr"], data["dlr"], data["b1"], data["b2"], data["n_cpu"],
          data["img_height"], data["img_width"], data["img_depth"], data["channels"], data["sample_interval"],
          data["checkpoint_interval"], data["lambda_voxel"], tensorboard_writer)
