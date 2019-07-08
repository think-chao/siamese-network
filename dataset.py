from torch.utils.data import Dataset
import os
import torchvision
import torch.nn as nn
from PIL import Image
import random
class HRDataset(Dataset):
    def __init__(self, data_path, transform = None):
        super(HRDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.all_image_name = self.get_all_image_name()

    def __len__(self):
        return len(self.all_image_name)

    def __getitem__(self, item):
        similar = 0

        img1 = self.all_image_name[item][0]
        img1_label = img1.split('/')[2]

        img2_index = random.randint(0, len(self.all_image_name)-1)
        img2 = self.all_image_name[img2_index][0]
        img2_label = img2.split('/')[2]

        if img2_label==img1_label:
            similar =1

        image1 = Image.open(img1).convert('L')
        image2 = Image.open(img2).convert('L')
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        # img_arr = [image1, image2]

        return {'Image1': image1, 'Image2': image2, 'Similar': similar}

    def get_all_image_name(self):
        dst = torchvision.datasets.ImageFolder(self.data_path)
        # print(dst.imgs)
        return dst.imgs
        # all_image_name = []
        # for dir in os.listdir(self.data_path):
        #     sub_dir = os.path.join(self.data_path, dir)
        #     for img in os.listdir(sub_dir):
        #         img_name = os.path.join(sub_dir, img)
        #         all_image_name.append(img_name)
        # return all_image_name


