import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils import data
import glob

class Cityscapes_SRC_Dataset(data.Dataset):

    def __init__(self, img_root, label_root, crop_size=(11, 11), ignore_label=255, mean=(128, 128, 128), set='train'):
        self.img_root = img_root  # folder for GTA5 which contains subfolder images, labels
        self.label_root = label_root
        self.set = set
        self.mean = mean
        self.crop_size = crop_size
        self.files = []
        self.X_path = []
        self.y_path = []
        self.ignore_label = ignore_label

        for num, name in enumerate(sorted(glob.glob(img_root + set + '/' + '/**/' + '*.png'))):
            self.X_path.append(name)

        for num, name in enumerate(sorted(glob.glob(label_root + set + '/' + '/**/' + '*labelIds.png'))):
            self.y_path.append(name)


    def __len__(self):
        return len(self.X_path)

    def __getitem__(self, index):

        image = Image.open(self.X_path[index]).convert('RGB')
        label = Image.open(self.y_path[index])
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size)

