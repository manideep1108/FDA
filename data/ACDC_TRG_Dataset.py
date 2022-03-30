import numpy as np
from PIL import Image
from torch.utils import data
import glob

class ACDC_TRG_Dataset(data.Dataset):

    def __init__(self, img_root, label_root, crop_size=(11, 11), mean=(128, 128, 128), set='train'):
        self.img_root = img_root  # folder for GTA5 which contains subfolder images, labels
        self.label_root = label_root
        self.set = set
        self.mean = mean
        self.crop_size = crop_size
        self.files = []
        self.X_path = []

        for num, name in enumerate(sorted(glob.glob(img_root + '/**/' + set + '/' + '/**/' + '*.png'))):
            self.X_path.append(name)


    def __len__(self):
        return len(self.X_path)

    def __getitem__(self, index):
        image = Image.open(self.X_path[index]).convert('RGB')
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size)

