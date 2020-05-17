""" Loads the ShapeNet SDF Data into a TorchDataset
Use download_shapenet.sh to download it.
"""
import os
import torch
import numpy as np
from torch.utils import data

class ShapeNetDataset(data.Dataset):
    def __init__(self, category="chairs", shapenet_dir="data/ShapeNet_SDF"):
        categories = ["airplanes", "chairs", "sofas"]
        assert category in categories, f"category={category} not supported, should be one of: {categories}"
        self.dir = os.path.join(shapenet_dir, category, "surface")


    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        """ Returns a 4 column torch tensor: [x, y, z, SDF(x, y, z)]
        """
        assert index >= 0 and index < self.__len__(), f"Index {index} should be between 0 and {self.__len__() - 1}"

        sample_path = os.path.join(self.dir, os.listdir(self.dir)[index])

        np_sample = np.load(sample_path)
        sample = torch.from_numpy(np_sample)

        return sample
