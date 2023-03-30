import torch
import torchvision
import random
from re import L
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image

from .base_dataloader import BaseDataLoader
from .base_scd_dataset import SCDBaseDataSet
totensor = torchvision.transforms.ToTensor()
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
A_ANNOT_FOLDER_NAME = "A_L"
B_ANNOT_FOLDER_NAME = "B_L"

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name, annot):
    return os.path.join(root_dir, annot, img_name) #.replace('.jpg', label_suffix)

num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Color2Index(ColorLabel):
    data = np.array(ColorLabel, dtype=np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def TensorIndex2Color(pred):
    colormap = torch.as_tensor(ST_COLORMAP, dtype=torch.uint8).to(pred.device)
    x = pred.long()
    return colormap[x, :]

def transform_augment_cd(img, split='val', min_max=(0, 1)):
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img


class ImageDataset(SCDBaseDataSet):
    def __init__(self, data_dir, split, augment=True,
                jitter=False, use_weak_lables=False, weak_labels_output=None, crop_size=None, scale=False, flip=False, rotate=False,
                blur=False, percnt_lbl=None, data_len=-1, **kwargs):
        self.num_classes = 7
        self.data_len = data_len
        self.percnt_lbl = percnt_lbl
        self.split = split
        super(ImageDataset, self).__init__(data_dir=data_dir, split=split, augment=augment,
                                           jitter=jitter, use_weak_lables=use_weak_lables, weak_labels_output=weak_labels_output,
                                           crop_size=crop_size, scale=scale, flip=flip, rotate=rotate, blur=blur,
                                            )
    
    def _set_files(self):
        if self.split == "val":
            file_list = os.path.join(self.root, 'list', f"{self.split}" + ".txt")
        elif self.split == "test":
            file_list = os.path.join(self.root, 'list', f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")
        img_name_list = np.loadtxt(file_list, dtype=str)
        if img_name_list.ndim == 2:
            return img_name_list[:, 0]
        self.dataset_len = len(img_name_list)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        self.files = img_name_list[:self.data_len]

    def _load_data(self, index):
        image_A_path    = os.path.join(self.root, 'A', self.files[index%self.data_len])
        image_B_path    = os.path.join(self.root, 'B', self.files[index%self.data_len])
        image_A         = np.asarray(Image.open(image_A_path), dtype=np.float32)
        image_B         = np.asarray(Image.open(image_B_path), dtype=np.float32)
        image_id        = self.files[index%self.data_len].split("/")[-1].split(".")[0]
        AL_path  = os.path.join(self.root, 'A_L', self.files[index%self.data_len])
        BL_path  = os.path.join(self.root, 'B_L', self.files[index%self.data_len])
        # 转成分类索引0,1,2,..,7
        img_lb_Al = Color2Index(Image.open(AL_path).convert("RGB"))
        img_lb_Bl = Color2Index(Image.open(BL_path).convert("RGB"))
        # label_bn = (image_A>0).astype(np.uint8)
        return image_A, image_B, img_lb_Al, img_lb_Bl, image_id
    
    

class SCDDataLoader(BaseDataLoader):
    def __init__(self, dataset, **kwargs):
        self.batch_size = kwargs.pop('batch_size')
        
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')
        
        self.dataset = dataset
        super().__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)
