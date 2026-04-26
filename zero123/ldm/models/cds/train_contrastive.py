import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from lightly.loss import NTXentLoss
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from lightly.utils.debug import std_of_l2_normalized
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
)

from model import FRUITS_I, complex_resnet18
import layers
import numpy as np
import pickle
from PIL import Image
from easydict import EasyDict
from pytorch_lightning import loggers as pl_loggers
import json


class SAMPLEData(torch.utils.data.Dataset):
  def __init__(self,
      root_dir="data/SAMPLE_dataset_public",
      db_file="db_g01_offset_1e-3.pkl",
      validation=False,
      transform=None,
      syn2real=False,
      is_regression=False,
  ):

    super().__init__()
    self.root_dir = root_dir
    self.db_file = db_file
    if transform is None:
      transform = lambda x: x
    self.transform = transform
    self.is_regression = is_regression
    self.syn2real = syn2real

    with open(os.path.join(root_dir, db_file), 'rb') as f:
      self.db = pickle.load(f)

    # find classes
    self.name2class = {}
    cls = 0
    for d in self.db:
      class_name = d['target_name'].item()
      if not class_name in self.name2class:
        self.name2class[class_name] = cls
        cls += 1
      d['class'] = self.name2class[class_name]

    total_objects = len(self.db)
    if syn2real:
      train_indices = [i for i in range(total_objects) if 'synth' in self.db[i]['fname']]
      valid_indices = [i for i in range(total_objects) if 'real' in self.db[i]['fname']]
      assert(len(train_indices) == len(valid_indices))
      assert(2*len(train_indices) == total_objects)
    else:
      rng = np.random.RandomState(seed=42)
      indices = rng.permutation(np.arange(total_objects))
      train_indices = indices[:-269]
      valid_indices = indices[-269:]

    if validation:
      self.db = [self.db[i] for i in valid_indices]
    else:
      self.db = [self.db[i] for i in train_indices]

    print('============= length of dataset %d =============' % len(self.db))
    print(f'   === train indices: {train_indices[:10]}  ===')
    print(f'   === valid indices: {valid_indices[:10]}  ===')

  def __len__(self):
    return len(self.db)

  def __getitem__(self, index):
    data = self.db[index]
    image = data['image']
    if self.is_regression:
      return self.transform(image), data['azimuth'].item() # azimuth in radian
    else:
      return self.transform(image), data['class']

def monotonic_transform(im, transform='id', Q=256):
  """Computes random monotonic augmentation of values.
  The min/max values of the image remain the same.
  Args:
    im: np.ndarray with integer values between [0-Q-1].
        usually Q=256 and we use np.uint8 for im
    transform: transformation on the offsets before mapping
    Q: quantization level
  """
  cs = np.cumsum([0] + np.random.rand(Q-1).tolist())
  mapping = cs/cs.max()
  if transform == 'id':
    pass
  if transform == 'sqrt':
    mapping = np.sqrt(mapping)
  elif transform == 'square':
    mapping = mapping**2
  f = lambda x: mapping[x]
  vfunc = np.vectorize(f)
  aug_im = (vfunc(im)*255.).astype(np.uint8)

def augment_magnitude(im, epsilon=0.1, noise_type='speckle',
                      transform='id', random_monotonic=True, Q=256):
  """
  Args:
    im: (np.int32) array with values between 0 and Q-1
  Returns:
    aug_im: (np.float32) array with values between 0 and Q-1
  """
  if random_monotonic:
    cs = np.cumsum(np.append([0], np.random.rand(Q-1)))
  else:
    cs = np.arange(Q)
  mapping = cs / cs.max()
  if epsilon > 0:
    if noise_type == 'gauss':
      mapping = mapping + np.random.randn(*mapping.shape) * epsilon
    elif noise_type == 'speckle':
      mapping = mapping + np.random.randn(*mapping.shape) * epsilon * mapping
    elif noise_type == 'id':
      pass
    else:
      raise ValueError(f"noise type {noise_type} is not defined!")
  mapping = (mapping - mapping.min())/(mapping.max() - mapping.min())
  if transform == 'id':
    pass
  if transform == 'sqrt':
    mapping = np.sqrt(mapping)
  elif transform == 'square':
    mapping = mapping**2
  f = lambda x: mapping[x]
  vfunc = np.vectorize(f)
  aug_im = vfunc(im)*(Q-1)
  return aug_im

def quantize(tensor, Q=256, min_to_use=None, max_to_use=None):
  """quantize a tensor to Q bins
  Args:
    tensor: (np.float32) input tensor
    Q: (int) quantization resolution
    min_to_use: (none) or (float) if none uses tensor min, otherwise
                uses the `min_to_use` instead of tensor min.
    max_to_use: (none) or (float) if none uses tensor max, otherwise
                uses the `max_to_use` instead of tensor max.

  Returns:
    quantized_tensor: (np.int32) tensor with values between 0 and Q-1
    tensor_min = (float) min value of tensor before quantization
    tensor_max = (float) max value of tensor before quantization
  """
  tensor_min = min_to_use or tensor.min()
  tensor_max = max_to_use or tensor.max()
  # normalize to 01
  tensor = (tensor - tensor_min)/(tensor_max - tensor_min)
  return np.int32(tensor*(Q-1)), tensor_min, tensor_max

class Augmentor:
  def __init__(self, epsilon=0.1, noise_type='speckle', transform='id', random_monotonic=True,
               is_complex=False, Q=256, complex_mode='real_imag_sep', normalize01=False,
               apply_log_on_mag=False):
    self.epsilon = epsilon
    self.noise_type = noise_type
    self.transform = transform
    self.random_monotonic = random_monotonic
    self.is_complex = is_complex
    self.Q = Q
    self.complex_mode = complex_mode
    self.normalize01 = normalize01
    self.apply_log_on_mag = apply_log_on_mag
    if apply_log_on_mag:
      assert(normalize01)
    if not is_complex:
      assert(Q == 256)

  def quantize_and_augment(self, tensor, max_to_use=None, min_to_use=None):
    tensor, tensor_min, tensor_max = quantize(tensor, Q=self.Q, max_to_use=max_to_use, min_to_use=min_to_use)
    aug_tensor = augment_magnitude(tensor, self.epsilon, self.noise_type, 'id', self.random_monotonic, Q=self.Q)
    aug_tensor = aug_tensor / (self.Q-1)*(tensor_max - tensor_min) + tensor_min
    return np.float32(aug_tensor)

  def __call__(self, im):
    if self.is_complex:
      if (self.epsilon == 0.0 or self.noise_type == 'id') and self.random_monotonic == False: # identity
        return np.stack([im.real[None], im.imag[None]], axis=0) # [2,1,H,W]
      if self.complex_mode == 'real_imag_sep':
        # Quantization to Q values for both real and imag
        #TODO think about changing min/max
        aug_real = self.quantize_and_augment(im.real)
        aug_imag = self.quantize_and_augment(im.imag)
        return np.stack([aug_real[None], aug_imag[None]], axis=0) # [2,1,H,W] tensor
      elif self.complex_mode == 'real_imag_same':
        aug_tensor = self.quantize_and_augment(np.stack([im.real, im.imag],axis=0)) # [2,H,W]
        return aug_tensor[:,None] # [2,1,H,W]
      elif self.complex_mode == 'mag_only':
        mag = np.abs(im)
        angle = np.angle(im) #[-pi, pi]
        # I want to change min and max randomly here (similar to the non complex case)
        # and using the fact that mag is normalized globally and it is between 0 and 1
        if self.normalize01:
          mag = np.log10(mag+1e-3) if self.apply_log_on_mag else mag
          aug_mag = self.quantize_and_augment(mag)
        else:
          aug_mag = self.quantize_and_augment(mag, max_to_use=1.0, min_to_use=0.0)
        real = aug_mag * np.cos(angle)
        imag = aug_mag * np.sin(angle)
        return np.stack([real[None], imag[None]], axis=0) # [2,1,H,W]
      elif self.complex_mode == 'mag_phase_sep':
        mag = np.abs(im)
        angle = np.angle(im) #[-pi, pi]
        if self.normalize01:
          mag = np.log10(mag+1e-3) if self.apply_log_on_mag else mag
          aug_mag = self.quantize_and_augment(mag)
          aug_angle = self.quantize_and_augment(angle)
        else:
          aug_mag = self.quantize_and_augment(mag, max_to_use=1.0, min_to_use=0.0)
          aug_angle = self.quantize_and_augment(angle, max_to_use=np.pi, min_to_use=-np.pi)
        real = aug_mag * np.cos(aug_angle)
        imag = aug_mag * np.sin(aug_angle)
        return np.stack([real[None], imag[None]], axis=0) # [2,1,H,W]
    else:
      # im is already uint8 and in [0,255]
      return Image.fromarray(np.uint8(augment_magnitude(im, self.epsilon, self.noise_type, 'id', self.random_monotonic, Q=self.Q)))


class MaskedContrastiveLossWithLabels(nn.Module):
  def __init__(
      self,
      temperature=0.5,
      gather_distributed=False,
      scale_negatives=True,
      eps=1e-7
  ):
    super().__init__()
    self.temperature = temperature
    self.gather_distributed = gather_distributed
    if gather_distributed:
      raise NotImplementedError(f"this loss is not implemented for distributed training")
    self.scale_negatives = scale_negatives
    self.eps = eps

  def forward(self, z0, z1, Y):
      """Masked version of cantrastive loss that uses labels tensor Y to create the mask.
      The negative examples having the same class as query are masked.
      If scale_negatives is True, It scales the logits as well based on the number of non-masked
      negative samples.

      Args:
        z0:
          Output projection of the first set of transformed images.
          Shape: (batch_size, embedding_size)

        z1:
          Output projection of the second set of transformed images.
          Shape: (batch_size, embedding_size)

        Y:
          Class labels.
          Shape: (batch_size)
      """
      device = z0.device
      B = z0.size(0) # batch size

      # normalize
      z0 = torch.nn.functional.normalize(z0, dim=1)
      z1 = torch.nn.functional.normalize(z1, dim=1)

      z00 = z0 @ z0.T / self.temperature
      z01 = z0 @ z1.T / self.temperature
      z10 = z01.T
      z11 = z1 @ z1.T / self.temperature

      z0100 = torch.cat([z01, z00], dim=1)
      z1011 = torch.cat([z10, z11], dim=1)

      # 2Bx2B logits
      logits = torch.cat([z0100, z1011], dim=0)

      # BxB labels
      expanded_labels = Y.view(B, 1).expand(B,B)
      # BxB mask: True where the labels are equal, False otherwise
      mask = torch.eq(expanded_labels, expanded_labels.t())
      # 2Bx2B mask like logits
      mask = mask.repeat(2,2)

      # labels for contrastive loss (01,10 part of logits)
      contrastive_labels = torch.arange(B,device=device).repeat(2)
      # make positive elements in the mask False (they have the same labels as themselves)
      #mask.scatter_(1, contrastive_labels.view(-1,1), False)
      mask[torch.arange(2*B, device=device), contrastive_labels] = False

      neg_mask = (~mask).float()

      # for stability
      logits -= logits[~mask].max()

      # prob = exp(logit_y) / (2*B/(\sum_j mask_j) * (\sum_j mask_j exp(logit_j))
      # log_prob = log(\sum_j mask_j / (2*B)) + logit_y - \
      #              logsumexp_j (logit_j + log(mask_j))

      log_prob = 0
      if self.scale_negatives:
        log_prob = torch.log(torch.sum(neg_mask, dim=1)/(2.0*B) + self.eps)

      log_prob += logits[torch.arange(2*B, device=device), contrastive_labels] # positives
      log_prob -= torch.logsumexp(logits + torch.log(neg_mask + self.eps), dim=1)
      loss = -log_prob.mean()
      return loss


class SimCLRModel(pl.LightningModule):
  def __init__(self, args):
    super().__init__()
    self.args = args
    if args.is_complex:
      resnet = complex_resnet18(num_classes=args.num_classes, is_complex=True, inp_channels=1)
    else:
      resnet = complex_resnet18(num_classes=args.num_classes, divisor=1., is_complex=False)
    hidden_dim = resnet.fc.in_features
    resnet.fc = torch.nn.Identity()
    self.backbone = resnet
    self.projection_head = heads.SimCLRProjectionHead(
        input_dim=hidden_dim, # TODO: 2 * prototype_size of CDS_I ??
        hidden_dim=hidden_dim,
        output_dim=128,
    )

    if args.use_masked_contrastive_loss:
      self.criterion = MaskedContrastiveLossWithLabels(
          temperature=0.5,
          gather_distributed=False,
          scale_negatives=True,
          eps=1e-7
      )
    else:
      self.criterion = NTXentLoss(gather_distributed=args.n_gpus > 1)

  def forward(self, x):
    features = self.backbone(x).flatten(start_dim=1)
    z = self.projection_head(features)
    return z

  def training_step(self, batch, batch_idx):
    (x0, x1), y, idx = batch
    z0 = self.forward(x0)
    z1 = self.forward(x1)
    if self.args.use_masked_contrastive_loss:
      loss = self.criterion(z0, z1, y)
    else:
      loss = self.criterion(z0, z1)
    self.log("train_loss_ssl", loss)
    with torch.no_grad():
      rep_std = std_of_l2_normalized(z0)
    self.log('representation_std', rep_std)
    return loss

  def configure_optimizers(self):
    if self.args.contrastive_lr is None:
      self.args.contrastive_lr = 6e-2
    optim = torch.optim.SGD(
        self.parameters(), lr=self.args.contrastive_lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.args.max_epochs)
    return [optim], [scheduler]


class Classifier(pl.LightningModule):
  def __init__(self, backbone, args):
    super().__init__()

    self.args = args
    self.backbone = backbone

    if args.fix_backbone:
      deactivate_requires_grad(backbone)
      self.fix_backbone_batchnorms()

    hidden_dim = 512
    if args.is_complex:
      hidden_dim = 726 #int(512/1.41)*2 TODO: fixed

    if args.is_regression:
      self.fc = nn.Linear(hidden_dim, 1) # only azimuth for now
      #self.fc = nn.Sequential(
      #    nn.Linear(hidden_dim, 128),
      #    nn.ReLU(),
      #    nn.Linear(128,1)
      #)
      self.criterion = nn.MSELoss()
    else:
      self.fc = nn.Linear(hidden_dim, args.num_classes)
      self.criterion = nn.CrossEntropyLoss()
    self.validation_step_outputs = []

  def fix_backbone_batchnorms(self):
    for module in self.backbone.modules():
      if isinstance(module, torch.nn.modules.BatchNorm1d):
        module.eval()
      if isinstance(module, torch.nn.modules.BatchNorm2d):
        module.eval()
      if isinstance(module, torch.nn.modules.BatchNorm3d):
        module.eval()


  def forward(self, x):
    y_hat = self.backbone(x).flatten(start_dim=1)
    y_hat = self.fc(y_hat)
    return y_hat

  def training_step(self, batch, batch_idx):
    if self.args.fix_backbone:
      self.fix_backbone_batchnorms()
    x,y = batch
    y_hat = self.forward(x)
    if self.args.is_regression:
      y = y.float()
      loss = self.criterion(y_hat[:,0], y)
    else:
      loss = self.criterion(y_hat, y)
    #print("\n\n\t\t ** Train \n\n");from IPython import embed;embed()
    self.log("train_loss_fc", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    if self.args.is_regression:
      num = y_hat.shape[0]
      mse_sum = torch.nn.functional.mse_loss(
          y_hat[:,0]*180/np.pi,
          y.float()*180/np.pi,
          reduction='sum'
      )
      self.validation_step_outputs.append((num, mse_sum))
      return num, mse_sum
    else:
      y_hat = torch.nn.functional.softmax(y_hat, dim=1)

      _, predicted = torch.max(y_hat, 1)
      num = predicted.shape[0]
      correct = (predicted == y).float().sum()
      self.validation_step_outputs.append((num, correct))
      return num, correct

  def on_validation_epoch_end(self):
    if self.validation_step_outputs:
      if self.args.is_regression:
        total_num = 0
        total_mse_sum = 0
        for num, mse_sum in self.validation_step_outputs:
          total_num += num
          total_mse_sum += mse_sum
        mse = total_mse_sum / total_num
        self.log('val_mse (deg)', mse, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
      else:
        total_num = 0
        total_correct = 0
        for num, correct in self.validation_step_outputs:
          total_num += num
          total_correct += correct
        acc = total_correct / total_num
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()

  def configure_optimizers(self):
    if self.args.fix_backbone:
      if self.args.classifier_lr is None:
        self.args.classifier_lr = 30.0
      optim = torch.optim.SGD(self.fc.parameters(), lr=self.args.classifier_lr)
    else:
      if self.args.classifier_lr is None:
        self.args.classifier_lr = 6e-2
      optim = torch.optim.SGD(self.parameters(), lr=self.args.classifier_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.args.max_epochs)
    return [optim], [scheduler]


def main(args):
  pl.seed_everything(args.seed)
  #FRUITS_I(prototype_size=128, dset_type='rgb')
  #backbone.dist_feat = torch.nn.Identity()

  model = SimCLRModel(args)


  if args.is_complex:
    view_trasform = torchvision.transforms.Compose(
        [
          Augmentor(
            epsilon=args.aug_epsilon, noise_type=args.aug_noise_type,
            random_monotonic=args.aug_random_monotonic, is_complex=args.is_complex,
            Q=args.aug_quantize_res, complex_mode=args.aug_complex_mode,
            normalize01=args.normalize01, apply_log_on_mag=args.apply_log_on_mag,
          ),
        ]
    )

    val_transform = torchvision.transforms.Lambda(
        lambda im: np.stack([im.real[None], im.imag[None]], axis=0)
    )
  else:
    view_trasform = torchvision.transforms.Compose(
        [
          Augmentor(
            epsilon=args.aug_epsilon, noise_type=args.aug_noise_type,
            random_monotonic=args.aug_random_monotonic, is_complex=args.is_complex,
            Q=args.aug_quantize_res, complex_mode=args.aug_complex_mode,
            normalize01=normalize01, apply_log_on_mag=args.apply_log_on_mag,
          ),
          torchvision.transforms.Resize((args.input_size, args.input_size)),
          torchvision.transforms.ToTensor(),
        ]
    )

    val_transform = torchvision.transforms.Compose(
        [
          torchvision.transforms.ToPILImage(),
          torchvision.transforms.Resize((args.input_size, args.input_size)),
          torchvision.transforms.ToTensor(),
        ]
    )

  transform = MultiViewTransform(transforms=[view_trasform, view_trasform])
  train_dataset_simclr = LightlyDataset.from_torch_dataset(
      SAMPLEData(db_file=args.db_file, is_regression=args.is_regression, syn2real=args.syn2real),
      transform=transform
  )

  # default was val_transform i.e, args.augment_classifier = False
  train_classifier_transform = view_trasform if args.augment_classifier else val_transform

  train_dataset_classifier = SAMPLEData(
      db_file=args.db_file,
      validation=False, transform=train_classifier_transform,
      syn2real=args.syn2real, is_regression=args.is_regression,
  )
  val_dataset = SAMPLEData(
      db_file=args.db_file,
      validation=True, transform=val_transform,
      syn2real=args.syn2real, is_regression=args.is_regression,
  )

  # Build a PyTorch dataloader.
  train_dataloader_simclr = torch.utils.data.DataLoader(
      train_dataset_simclr,  # Pass the dataset to the dataloader.
      batch_size=args.batch_size,  # A large batch size helps with the learning.
      shuffle=True,  # Shuffling is important!
      drop_last=True,
      num_workers=args.num_workers,
  )

  train_dataloader_classifier = torch.utils.data.DataLoader(
      train_dataset_classifier,
      batch_size=args.batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=args.num_workers,
  )

  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      drop_last=False,
      num_workers=args.num_workers,
  )

  if not args.classifier_only:
    lr_monitor = LearningRateMonitor(logging_interval='step')
    log_folder="lightning_logs/contrastive"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_folder, name=args.experiment_name)
    if args.n_gpus > 1:
      trainer = pl.Trainer(
          logger=tb_logger, max_epochs=args.max_epochs, devices=args.n_gpus, accelerator="gpu",
          log_every_n_steps=2, strategy="ddp", sync_batchnorm=True, use_distributed_sampler=True,
          callbacks=[lr_monitor],
      )
    else:
      trainer = pl.Trainer(
          logger=tb_logger, max_epochs=args.max_epochs, devices=1, accelerator="gpu", log_every_n_steps=2,
          callbacks=[lr_monitor],
      )

    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    with open(os.path.join(trainer.logger.log_dir, 'args.json'), 'w') as f:
      json.dump(args, f, indent=2)
    trainer.fit(model, train_dataloader_simclr)
  else:
    if args.resume_ckpt is not None:
      print(f"Loading from {args.resume_ckpt} ...")
      ckpt = torch.load(args.resume_ckpt, map_location='cpu')
      model.load_state_dict(ckpt['state_dict'])
      print('Done!')
    else:
      print("WARNING: model is not loading any checkpoint")


  model.eval()
  classifier = Classifier(model.backbone, args)
  log_folder = "lightning_logs/classifier"
  lr_monitor = LearningRateMonitor(logging_interval='step')
  if args.is_regression:
    log_folder = "lightning_logs/regression"
  tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_folder, name=args.experiment_name)
  if args.n_gpus > 1:
    trainer = pl.Trainer(
        logger=tb_logger, max_epochs=args.max_epochs, devices=args.n_gpus, accelerator="gpu", log_every_n_steps=2,
        strategy="ddp", sync_batchnorm=True, use_distributed_sampler=True, callbacks=[lr_monitor]
    )
  else:
    trainer = pl.Trainer(
        logger=tb_logger, max_epochs=args.max_epochs, devices=1, accelerator="gpu", log_every_n_steps=2,
        callbacks=[lr_monitor]
    )

  os.makedirs(trainer.logger.log_dir, exist_ok=True)
  with open(os.path.join(trainer.logger.log_dir, 'args.json'), 'w') as f:
      json.dump(vars(args), f, indent=2)
  trainer.fit(classifier, train_dataloader_classifier, val_loader)

if __name__ == '__main__':
  torch.set_float32_matmul_precision('high')
  #args = EasyDict(
    #{
      #'experiment_name': 'masked_fix_bn',
      #'seed': 42,
      #'num_workers': 1,
      #'batch_size': 256,
      #'max_epochs': 500,
      #'input_size': 128,
      #'n_gpus': 1,
      #'num_classes': 10,
      #'classifier_only': True,
      #'fix_backbone': True,
      #'resume_ckpt': "/home/amirr/codes/cds/lightning_logs/contrastive/masked_loss/version_0/checkpoints/epoch=499-step=4500.ckpt", #None,
      ##'resume_ckpt': "/home/amirr/codes/cds/lightning_logs/contrastive/both_e500/version_1/checkpoints/epoch=499-step=4500.ckpt", #None,
      ##'resume_ckpt': "/home/amirr/codes/cds/lightning_logs/contrastive/masked_loss/version_0/checkpoints/epoch=499-step=4500.ckpt", #None,
      #'use_masked_contrastive_loss':  True,
      #'classifier_lr': 0.006, # for regression use 0.006, for classification use 30.0 when backbone is fixed
      #'contrastive_lr': None,
      #'is_regression': True,
      #'syn2real': False,
      #'aug_epsilon': 0.1,
      #'aug_noise_type': 'speckle',
      #'aug_random_monotonic': True,
      #'augment_classifier': False,
    #}
  #)

  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment_name", type=str, default="experiment_name")
  parser.add_argument("--seed", type=int, default=42, help="global random seed.")
  parser.add_argument("--num_workers", type=int, default=1, help="number of workers for dataloader.")
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--max_epochs", type=int, default=500)
  parser.add_argument("--input_size", type=int, default=128)
  parser.add_argument("--n_gpus", type=int, default=1)
  parser.add_argument("--num_classes", type=int, default=10)
  parser.add_argument("--classifier_only", action='store_true', default=False)
  parser.add_argument("--fix_backbone", action='store_true', default=False)
  parser.add_argument("--resume_ckpt", type=str, default=None)
  parser.add_argument("--use_masked_contrastive_loss", action='store_true', default=False)
  parser.add_argument("--classifier_lr", type=float, default=None)
  parser.add_argument("--contrastive_lr", type=float, default=None)
  parser.add_argument("--is_regression", action='store_true', default=False)
  parser.add_argument("--syn2real", action='store_true', default=False)
  parser.add_argument("--aug_epsilon", type=float, default=0.1)
  parser.add_argument("--aug_noise_type", type=str, default='speckle')
  parser.add_argument("--aug_random_monotonic", action='store_true', default=False)
  parser.add_argument("--augment_classifier", action='store_true', default=False)
  parser.add_argument("--is_complex", action='store_true', default=False)
  parser.add_argument("--db_file", type=str, default="db_g01_offset_1e-3.pkl")
  parser.add_argument("--aug_quantize_res", type=int, default=256)
  parser.add_argument("--aug_complex_mode", type=str, default="real_imag_sep",
                      choices=["real_imag_sep",
                               "real_imag_same",
                               "mag_only",
                               "mag_phase_sep",
                      ])
  parser.add_argument("--normalize01", action='store_true', default=False)
  parser.add_argument("--apply_log_on_mag", action='store_true', default=False)

  parser.set_defaults()
  args = parser.parse_args()

  main(args)
