from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
import ldm.data.pyramid_utils as pyramid_utils
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
import glob
import scipy
import pickle

# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)

def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [NfpDataset(x, image_transforms=copy.copy(tforms), default_caption="A view from a train window") for x in dirs]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2*self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i+1)*self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES,prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[...,::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[...,::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption
        }
        return data

# end hacky things


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [FolderData(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)



class NfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1


    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index+1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        try:
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        except:
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1') # this one we know is valid
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


class SAMPLEAugmentor:
  def __init__(self, epsilon=0.1, noise_type='speckle', transform='id'):
    self.epsilon = epsilon
    self.noise_type = noise_type
    self.transform = transform

  def __call__(self, im):
    cs = np.cumsum([0] + np.random.rand(255).tolist())
    mapping = cs / cs.max()
    if self.epsilon > 0:
      if self.noise_type == 'gauss':
        mapping = mapping + np.random.randn(*mapping.shape) * self.epsilon
      elif self.noise_type == 'speckle':
        mapping = mapping + np.random.randn(*mapping.shape) * self.epsilon * mapping
      mapping = (mapping - mapping.min())/(mapping.max() - mapping.min())
    if self.transform == 'id':
      pass
    if self.transform == 'sqrt':
      mapping = np.sqrt(mapping)
    elif self.transform == 'square':
      mapping = mapping**2
    f = lambda x: mapping[x]
    vfunc = np.vectorize(f)
    aug_im = (vfunc(im)*255.).astype(np.uint8)
    return Image.fromarray(aug_im)


class SAMPLEDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, db_file, batch_size, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.db_file = db_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        if 'image_transforms' in validation:
            val_image_transforms = [torchvision.transforms.Resize(validation.image_transforms.size)]
        else:
            val_image_transforms = []
        val_image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.val_image_transforms = torchvision.transforms.Compose(val_image_transforms)

        train_image_transforms = []
        if 'image_transforms' in train:
          if 'augmentation' in train.image_transforms:
            print("Instantiating SAMPLEAugmentor ...")
            train_image_transforms += [instantiate_from_config(train.image_transforms.augmentation)]
          train_image_transforms += [torchvision.transforms.Resize(train.image_transforms.size)]
        train_image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.train_image_transforms = torchvision.transforms.Compose(train_image_transforms)

    def train_dataloader(self):
        dataset = SAMPLEData(root_dir=self.root_dir, db_file=self.db_file, validation=False, \
                                image_transforms=self.train_image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = SAMPLEData(root_dir=self.root_dir, db_file=self.db_file, validation=True, \
                                image_transforms=self.val_image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return wds.WebLoader(SAMPLEData(root_dir=self.root_dir, db_file=self.db_file,
                             validation=self.validation, image_transforms=self.val_image_transforms), \
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class SAMPLEData(Dataset):
    def __init__(self,
        root_dir='data/SAMPLE_dataset_public',
        db_file="db.pkl",
        image_transforms=[],
        ext="mat",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.db_file = db_file
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(root_dir, db_file), 'rb') as f:
            self.db = pickle.load(f)


        total_objects = len(self.db)
        if validation:
            #self.db = self.db[math.floor(total_objects / 100. * 96.):] # used last 4% as validation
            self.db = self.db[-92:] # used last 92 for validation
        else:
            #self.db = self.db[:math.floor(total_objects / 100. * 96.)] # used first 96% as training
            self.db = self.db[:-92] # used remaining for training
        print('============= length of dataset %d =============' % len(self.db))
        self.tform = image_transforms

    def __len__(self):
        return len(self.db)

    def get_T(self, target_data, cond_data):
        theta_cond, azimuth_cond, z_cond = cond_data['elevation'], cond_data['azimuth'], 0.0
        theta_target, azimuth_target, z_target = target_data['elevation'], target_data['azimuth'], 0.0

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) # TODO % (2 * math.pi) I am removing this
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z])
        return d_T

    def __getitem__(self, index):
        data = {}

        cond_data = self.db[index]
        target_data = self.db[random.choice(cond_data['neighbour_indices'])]

        if self.return_paths:
            data["path"] = str(cond_data['fname'])


        target_im = self.process_im(target_data)
        cond_im = self.process_im(cond_data)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_data, cond_data)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, data):
        img = Image.fromarray(data['image'])
        img = img.convert("RGB")
        return self.tform(img)


def mapping_based_augment(
    im, epsilon=0.1, noise_type='speckle', monotonic=True,
    p=1.0, Q=256, mono_e_min=1.0, mono_e_max=1.0):
  # Want to keep mapping: [0,1] -> [0,1]
  #if np.random.random_sample() >= p:
  #  return np.float32(im)

  if monotonic:
    cs = np.cumsum(np.append([0], np.random.rand(Q-1)))
  else:
    cs = np.arange(Q)

  ys = cs/cs.max()
  if noise_type == 'gauss':
    ys = ys + np.random.randn(*ys.shape) * epsilon
  elif noise_type == 'speckle':
    ys = ys + np.random.randn(*ys.shape) * epsilon * ys
  elif noise_type == 'speckle_rayleigh':
    ys = ys + np.random.rayleigh(epsilon, ys.shape) * ys
  elif noise_type == 'none':
    pass
  else:
    raise ValueError(f"noise_type {noise_type} is not defined!")

  ys = (ys - ys.min())/(ys.max() - ys.min())
  if mono_e_max != 1 or mono_e_min != 1:
    assert(mono_e_max >= mono_e_min and mono_e_min >= 0)
    # uniformly sampling between [e_min,1] and [1,e_max]
    if mono_e_min != mono_e_max and mono_e_min <= 1 and mono_e_max >= 1:
      u = np.random.rand()
      mn = 2*u*(1 - mono_e_min) + mono_e_min
      mx = 2*(u-0.5)*(mono_e_max-1) + 1
      exp = mn if u < 0.5 else mx
    else:
      exp = np.random.uniform(mono_e_min, mono_e_max)
    ys = ys ** exp
  xs = np.arange(Q) / (Q-1)
  im = np.interp(im, xs, ys)
  return np.float32(im)

def im2torch(im):
  im = im[None] #np.stack([im,im,im], axis=0)
  im = F.interpolate(torch.from_numpy(im)[None], (256,256), mode='bilinear', align_corners=True).float() #1,1,128,128
  return im

def compose_4_images(im, other_imgs, use_origs=None):
  """im: torch.float32 im
     other_imgs: List(torch.float32) images
  """
  assert(len(other_imgs) == 4)
  if use_origs is not None:
    assert(len(use_origs) == 4)
  _,_,H,W = im.shape
  use_origs = use_origs or random.choices([True, False], k=4)
  masks = [torch.zeros_like(im) for _ in range(4)]
  masks[0][:,:,:H//2,:W//2] = 1
  masks[1][:,:,:H//2,W//2:] = 1
  masks[2][:,:,H//2:,W//2:] = 1
  masks[3][:,:,H//2:,:W//2] = 1
  imgs = [im if use_orig else other for use_orig,other in zip(use_origs, other_imgs)]

  mask_pyrs = [pyramid_utils.laplacian_pyramid(m) for m in masks]
  imgs_pyrs = [pyramid_utils.laplacian_pyramid(img) for img in imgs]

  downC = mask_pyrs[0][1][-1]*imgs_pyrs[0][1][-1] + \
          mask_pyrs[1][1][-1]*imgs_pyrs[1][1][-1] + \
          mask_pyrs[2][1][-1]*imgs_pyrs[2][1][-1] + \
          mask_pyrs[3][1][-1]*imgs_pyrs[3][1][-1]

  lap_pyrC = [m0*i0 + m1*i1 + m2*i2 + m3*i3 for (i0,i1,i2,i3,m0,m1,m2,m3) in zip(
                  imgs_pyrs[0][0], imgs_pyrs[1][0], imgs_pyrs[2][0], imgs_pyrs[3][0],
                  mask_pyrs[0][1], mask_pyrs[1][1], mask_pyrs[2][1], mask_pyrs[3][1]
                )
  ]

  img_rC = pyramid_utils.reconstruct(lap_pyrC, downC)
  img_rC.clip_(0,1)

  return img_rC


class ShipAugmentor:
  def __init__(self, epsilon=0.0, noise_type='none', random_monotonic=True,
      Q=4096, p=0.95, is_decibel=True, global_min=-60.0, global_max=81.96689,
      mono_e_min=1.0, mono_e_max=1.0
      ):
    self.epsilon = epsilon
    self.noise_type = noise_type
    self.random_monotonic = random_monotonic
    self.Q = Q
    self.p = p
    self.is_decibel = is_decibel
    self.global_min = global_min
    self.global_max = global_max
    self.mono_e_min = mono_e_min
    self.mono_e_max = mono_e_max

  def __call__(self, im):
    if np.random.random_sample() >= self.p:
      return np.float32(im)
    if self.is_decibel:
      im = im * (self.global_max - self.global_min) + self.global_min
      im = 10**(im/20) - 1e-3
      orig_max = 10**(self.global_max/20) - 1e-3
      orig_min = 10**(self.global_min/20) - 1e-3
      im = (im - orig_min) / (orig_max - orig_min)
      im = mapping_based_augment(
          im, self.epsilon, self.noise_type, self.random_monotonic,
          self.p, self.Q, self.mono_e_min, self.mono_e_max,
      )
      np.clip(im, 0, 1)
      im = 20*np.log10(im+1e-3)
      im = (im - self.global_min)/(self.global_max-self.global_min)
      np.clip(im, 0, 1)
      return im
    else:
      return mapping_based_augment(
          im, self.epsilon, self.noise_type, self.random_monotonic,
          self.p, self.Q, self.mono_e_min, self.mono_e_max,
      )


class ShipDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, db_file, batch_size, lap_cutmix=0, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.db_file = db_file
        self.lap_cutmix = lap_cutmix
        self.batch_size = batch_size
        self.num_workers = num_workers

        default_transforms = [transforms.Lambda(lambda im: np.stack([im,im,im], axis=0)),
            transforms.Lambda(lambda im: torch.from_numpy(im))]


        if 'image_transforms' in validation:
            im_size = validation.image_transforms.size
            val_image_transforms = default_transforms + [transforms.Resize((im_size,im_size))]
        else:
            val_image_transforms = default_transforms
        val_image_transforms.extend([
          transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))]
        )
        self.val_image_transforms = torchvision.transforms.Compose(val_image_transforms)

        train_image_transforms = default_transforms
        if 'image_transforms' in train:
          if 'augmentation' in train.image_transforms:
            print("Instantiating ShipAugmentor ...")
            train_image_transforms = [instantiate_from_config(train.image_transforms.augmentation)] + default_transforms
          im_size = train.image_transforms.size
          train_image_transforms += [torchvision.transforms.Resize((im_size,im_size))]
        train_image_transforms.extend([
          transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))]
        )
        self.train_image_transforms = torchvision.transforms.Compose(train_image_transforms)

    def train_dataloader(self):
        dataset = ShipData(root_dir=self.root_dir, db_file=self.db_file, lap_cutmix=self.lap_cutmix, validation=False, \
                                image_transforms=self.train_image_transforms, part='train')
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ShipData(root_dir=self.root_dir, db_file=self.db_file, validation=True, \
                                image_transforms=self.val_image_transforms, part='val')
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return wds.WebLoader(ShipData(root_dir=self.root_dir, db_file=self.db_file,
                             validation=self.validation, image_transforms=self.val_image_transforms, part='val'),
                             batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ShipData(Dataset):
    def __init__(self,
        root_dir='data/SAMPLE_dataset_public',
        db_file="ship_g01.pkl",
        lap_cutmix=False,
        image_transforms=[],
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        validation=False,
        part='train'
        ) -> None:

        self.root_dir = Path(root_dir)
        self.db_file = db_file
        self.lap_cutmix = lap_cutmix
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess

        with open(os.path.join(root_dir, db_file), 'rb') as f:
            self.db = pickle.load(f)

        self.part = part
        total_objects = len(self.db[part])
        print('============= length of dataset %d =============' % len(self.db[part]))
        self.tform = image_transforms

    def __len__(self):
        return len(self.db[self.part])

    def get_T(self, target_data, cond_data):
        theta_cond, azimuth_cond, z_cond = cond_data['elevation'], cond_data['azimuth'], 0.0
        theta_target, azimuth_target, z_target = target_data['elevation'], target_data['azimuth'], 0.0

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) # TODO % (2 * math.pi) I am removing this
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z])
        return d_T

    def __getitem__(self, index):
        data = {}

        cond_data = self.db[self.part][index]
        target_data = self.db[self.part][random.choice(cond_data['neighbour_indices'])]

        if self.return_paths:
            data["path"] = str(cond_data['index'])


        target_im, cond_im = self.apply_lap_cutmix(target_data, cond_data)
        target_im = self.process_im(target_im)
        cond_im = self.process_im(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_data, cond_data)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def apply_lap_cutmix(self, target_data, cond_data):
      target_im = target_data['image']
      cond_im = cond_data['image']
      if self.lap_cutmix > random.random():
        assert('cls_neis' in cond_data)
        assert(self.part == 'train')
        classes_pool = list(set(cond_data['cls_neis'].keys()).intersection(set(target_data['cls_neis'].keys())))
        if len(classes_pool) > 0:
          cond_im = im2torch(cond_im) # 1,1,H,W
          target_im = im2torch(target_im)
          other_im_cls_ids = random.choices(classes_pool, k=4) # with replacement
          cond_other_imgs = [im2torch(self.db[self.part][cond_data['cls_neis'][c]]['image']) for c in other_im_cls_ids]
          target_other_imgs = [im2torch(self.db[self.part][target_data['cls_neis'][c]]['image']) for c in other_im_cls_ids]
          use_origs = random.choices([True, False], k=4)
          cond_im = compose_4_images(cond_im, cond_other_imgs, use_origs=use_origs)[0,0].numpy() #H,W
          target_im = compose_4_images(target_im, target_other_imgs, use_origs=use_origs)[0,0].numpy() #H,W

      return target_im, cond_im

    def process_im(self, im):
        return self.tform(im)


class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir/chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
import random

class TransformDataset():
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }


    def __getitem__(self, index):
        data = self.ds[index]

        im = data['image']
        im = im.permute(2,0,1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1,2,0)

        data['image'] = im
        data['txt'] = data['txt'] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]



import random
import json
class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir/retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data
