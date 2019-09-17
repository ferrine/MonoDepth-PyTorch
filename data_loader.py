import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, Subset


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        left_dir = os.path.join(root_dir, "image_02/data/")
        self.left_paths = sorted(
            [os.path.join(left_dir, fname) for fname in os.listdir(left_dir)]
        )
        if mode == "train":
            right_dir = os.path.join(root_dir, "image_03/data/")
            self.right_paths = sorted(
                [os.path.join(right_dir, fname) for fname in os.listdir(right_dir)]
            )
            assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.mode == "train":
            right_image = Image.open(self.right_paths[idx])
            sample = {"left_image": left_image, "right_image": right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image


class OrthancData(Dataset):
    def __init__(self, roor_dir, mode, transform=None, labels=(False, True)):
        import orthanc

        reader = orthanc.dataset.HDF5StereoReader(roor_dir, 1, num_frames=1)
        indices = np.full_like(reader.labels, False)
        for label in labels:
            indices |= reader.labels == label
        pos, = indices.nonzero()
        self.mode = mode
        self.transform = transform
        self.reader = Subset(reader, pos)

    def __getitem__(self, item):
        stereo, _ = self.reader[item]
        left_image, right_image = stereo[0]
        left_image, right_image = (
            Image.fromarray(left_image),
            Image.fromarray(right_image),
        )
        if self.mode == "train":
            sample = {"left_image": left_image, "right_image": right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image

    def __len__(self):
        return len(self.reader)
