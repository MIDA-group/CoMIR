# -*- coding: utf-8 -*-

# Python Standard Libraries
from glob import glob
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Deep Learning libraries
import torch
from torch.utils.data import Dataset

# Other libraries
# ~ Scientific
import numpy as np
# ~ Image manipulation / visualisation
import imgaug
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import skimage.io as skio
import itertools
import skimage
# ~ Other
from tqdm import tqdm
import cv2, shutil

# %%

# %%
def pad_sample(img, d):
    # pad the image size to multiple of divisor d
    (w, h) = img.shape[:2]
    w_pad = (w // d + 1) * d - w
    h_pad = (h // d + 1) * d - h            
    wl = w_pad // 2
    wr = w_pad - wl
    hl = h_pad // 2
    hr = h_pad - hl
    if img.ndim == 2:
        img_pad = np.pad(img, ((wl, wr), (hl, hr)), 'constant', constant_values=0)
    else:
        img_pad = np.pad(img, ((wl, wr), (hl, hr), (0, 0)), 'constant', constant_values=0)
    return img_pad

class OverSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, steps_per_epoch):
        self.data_source = data_source
        self.steps_per_epoch = steps_per_epoch

        if not isinstance(steps_per_epoch, int) or self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch should be a positive integer "
                             "value, but got num_samples={}".format(self.steps_per_epoch))

    @property
    def num_samples(self):
        return self.steps_per_epoch

    def __iter__(self):
        n = len(self.data_source)
        return iter(np.random.choice(np.arange(n), self.steps_per_epoch, replace=True))

    def __len__(self):
        return self.steps_per_epoch
    
class SlideDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        if not isinstance(folder_path, list):
            folder_path = [folder_path]
        self.path = folder_path
        self.filenames = [glob(path) for path in folder_path]
        self.filenames = list(itertools.chain(*self.filenames))

        dataset = {}
        for pathname in tqdm(self.filenames):
            filename = os.path.basename(pathname)
            file_id = "_".join(filename.split("_")[:2])
            file_type = filename.split('_')[-1].split('.')[0]
            
            if file_id not in dataset.keys():
                dataset[file_id] = {}

            img = skio.imread(pathname)
            img = skimage.img_as_float(img)

            if file_type == "SHG":
                img = np.log(1.+img)
            if img.ndim == 2:
                img = img[..., np.newaxis]
            dataset[file_id][file_type] = img

        self.images = []
        for image_set in dataset:
            try:
                self.images.append(
                    np.block([
                        dataset[image_set]["SHG"],
                        dataset[image_set]["BF"]
                    ]).astype(np.float32)
                )
            except ValueError:
                print(f"Failed concatenating set {image_set}."
                      + f"Shapes are {dataset[image_set]['SHG'].shape} and {dataset[image_set]['BF'].shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx, augment=True):
        if augment and self.transform:
            return self.transform(self.images[idx])
        return self.images[idx]

class ImgAugTransform:
    def __init__(self, testing=False):
        if not testing:
            self.aug = iaa.Sequential([
                iaa.CropToFixedSize(256,256),
#                iaa.size.Resize(128, interpolation='linear'),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-180, 180), order=[0, 1, 3], mode="symmetric"),
                iaa.Sometimes(0.2, iaa.OneOf([
                    #iaa.AdditiveGaussianNoise(loc=0, scale=(0., 0.05)),
                    iaa.GaussianBlur(sigma=(0, 1.0)),
                    #iaa.EdgeDetect(alpha=1.0),
                    #iaa.CoarseDropout(0.1, size_percent=0.05, per_channel=True),
                ])),
                #iaa.Multiply((0.9, 1.1), per_channel=0.3),
            ])
        else:
            self.aug = iaa.Sequential([
                iaa.CropToFixedSize(128,128),
            ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


# %%
def make_EliceiriP2P_train(src_dir, target_dir):
    '''
    Create folder `/path/to/data` with subfolders `A` and `B`. 
    `A` and `B` should each have their own subfolders `train`, `val`, `test`, etc. 
    In `/path/to/data/A/train`, put training images in style A. 
    In `/path/to/data/B/train`, put the corresponding images in style B. 
    '''
#    src_dir='./Datasets/HighRes_Splits'
#    target_dir='./Datasets/Eliceiri_temp'
#    is_test=False
    
    data_folder = 'CNN_Training_Crops'
    
    dset = SlideDataset(
            [f'{src_dir}/{data_folder}/*_SHG.tif', 
             f'{src_dir}/{data_folder}/*_BF.tif'], 
            transform=ImgAugTransform())
#    dset_test = ZurichDataset([], transform=ImgAugTransform(testing=is_test))
    
    for modality in ['A', 'B']:
        for folder in ['train', 'val', 'test']:
            if not os.path.exists(f'{target_dir}/{modality}/{folder}'):
                os.makedirs(f'{target_dir}/{modality}/{folder}')
    
    epochs = 10
    batch_size = 16
    steps_per_epoch = 32
    samples_per_epoch = steps_per_epoch * batch_size
    
    # DATASET RELATED
    def worker_init_fn(worker_id):
        base_seed = int(torch.randint(2**32, (1,)).item())
        lib_seed = (base_seed + worker_id) % (2**32)
        imgaug.seed(lib_seed)
        np.random.seed(lib_seed)
    
    dataloader_args = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True,
        "worker_init_fn": worker_init_fn,
    }
    
    loader = torch.utils.data.DataLoader(
        dset,
        sampler=OverSampler(dset, samples_per_epoch),
        **dataloader_args
    )

    for epoch in tqdm(range(epochs)):
        for batch_idx, data in enumerate(loader): # 32 batches
            imgaug.seed(batch_idx)
            ABs = data.numpy() # data.shape = (16, 128, 128, 4)
            ABs = cv2.normalize(src=ABs, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            As = ABs[..., 0]
            Bs = ABs[..., 1:]
            for j in range(batch_size):
                skio.imsave(f'{target_dir}/A/train/e{epoch}_b{batch_idx}_{j}.tiff', 
                            As[j])
                skio.imsave(f'{target_dir}/B/train/e{epoch}_b{batch_idx}_{j}.tiff', 
                            Bs[j])
    return

# %%
make_EliceiriP2P_train(
        src_dir='./Datasets/HighRes_Splits',
        target_dir='./Datasets/Eliceiri_temp',
        is_test=False)

# %%
def make_EliceiriP2P_test(test_root, target_root):
    '''
    Create folder `/path/to/data` with subfolders `A` and `B`. 
    `A` and `B` should each have their own subfolders `train`, `val`, `test`, etc. 
    In `/path/to/data/A/train`, put training images in style A. 
    In `/path/to/data/B/train`, put the corresponding images in style B. 
    '''
#    test_root='./Datasets/HighRes_Splits/Registration_UltimateTesting'
#    target_root='./Datasets/Eliceiri_test'
    
    modalities = {'SHG':'A', 'BF':'B'}
    
    for test_dir in glob(f'{test_root}/*Patches'):
        exp = os.path.basename(test_dir).split('_')[0]
        target_dir = f'{target_root}/Eliceiri_{exp}'
        
        for modality in ['A', 'B']:
            for folder in ['train', 'val', 'test']:
                if not os.path.exists(f'{target_dir}/{modality}/{folder}'):
                    os.makedirs(f'{target_dir}/{modality}/{folder}')
    
        for img_path in tqdm(glob(f'{test_dir}/*')):
            pass
            file_name = os.path.basename(img_path)
            file_modality = file_name.split('_')[-2].split('.')[0]
            file_newname = file_name.replace(f'_{file_modality}_', '_')
            shutil.copyfile(img_path, f'{target_dir}/{modalities[file_modality]}/test/{file_newname}')
            
    return

# %%
make_EliceiriP2P_test(
        test_root='./Datasets/HighRes_Splits/Registration_UltimateTesting', 
        target_root='./Datasets/Eliceiri_test')