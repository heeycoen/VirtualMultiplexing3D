from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import glob


class CTDataset(Dataset):
    def __init__(self, datapath,onechan=False):
        self.datapath = datapath
        self.samples = glob.glob(self.datapath + '/*.h5')
        self.onechan = onechan

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dat = h5py.File(self.samples[idx], 'r')

        image = np.array([dat["mixed"][0]]) if self.onechan else np.array(dat["mixed"])
        unmixed = np.array(dat["unmixed"])

        return {"A": image, "B": unmixed}

class PredictDataset(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.samples = glob.glob(self.datapath + '/*.h5')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dat = h5py.File(self.samples[idx], 'r')

        image = np.array(dat["mixed"])

        return {"A": image}
