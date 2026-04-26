import torch
import torch.nn.functional as F
from torchvision.io import read_image
import numpy as np
import torchvision.transforms.functional as VF


def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
  kernel = torch.tensor([[1., 4., 6., 4., 1],
                         [4., 16., 24., 16., 4.],
                         [6., 24., 36., 24., 6.],
                         [4., 16., 24., 16., 4.],
                         [1., 4., 6., 4., 1.]], device=device)
  kernel /= 256.
  kernel = kernel.repeat(channels, 1, 1, 1)
  return kernel

def downsample(x):
  _,_,h,w = x.shape
  return F.interpolate(x, size=(h//2, w//2), mode='bilinear', align_corners=True)

def upsample_old(x):
  cc = torch.cat([x, torch.zeros_like(x)], dim=3)
  cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
  cc = cc.permute(0,1,3,2)
  cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
  cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
  x_up = cc.permute(0,1,3,2)
  return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def upsample(x, blur=False):
  _,_,h,w = x.shape
  x = F.interpolate(x, size=(h*2,w*2), mode='bilinear', align_corners=True)
  return conv_gauss(x, gauss_kernel(channels=x.shape[1], device=x.device)) if blur else x

def conv_gauss(img, kernel):
  img = F.pad(img, (2, 2, 2, 2), mode='reflect')
  out = F.conv2d(img, kernel, groups=img.shape[1])
  return out

def laplacian_pyramid(img, max_levels=3):
  current = img
  lap_pyr = []
  gauss_pyr = [img]

  kernel = gauss_kernel(channels=img.shape[1], device=img.device)

  for level in range(max_levels):
    filtered = conv_gauss(current, kernel)
    down = downsample(filtered)
    gauss_pyr.append(down)
    up = upsample(down)
    diff = current-up
    lap_pyr.append(diff)
    current = down
  return lap_pyr, gauss_pyr

def reconstruct(pyr, down):
  for lap in reversed(pyr):
    down = upsample(down) + lap
  return down

