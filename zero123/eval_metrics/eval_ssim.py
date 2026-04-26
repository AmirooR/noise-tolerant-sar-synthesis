import os
import numpy as np
from easydict import EasyDict
import pytorch_lightning as pl
import matplotlib
matplotlib.use('pdf')
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import json
from tqdm import tqdm

import scipy.linalg
import skimage
from datetime import datetime

def evaluate_ssim(args):
  print(f"targets folder: {args.viz_folder}")
  pl.seed_everything(1357)

  total_ssims = []
  total_azims = []

  print(f"cfg_scale is {args.cfg_scale}")

  for gen_seed in args.gen_seeds:
    print(f"running seed {gen_seed}")
    if 'viz2' in args.viz_folder:
      ext = f'random_{gen_seed}/random_{gen_seed}'
    else:
      ext = f'random_{gen_seed}'
    ssims = []
    for i in tqdm(range(args.num_samples)):
      for j in range(args.num_neighbours):
        target_file = os.path.join(args.viz_folder, ext, f'target_{i}_nei_{j}.png')
        sample_file = os.path.join(args.viz_folder, ext, f'pred_{i}_nei_{j}_sample_0_samples')
        if args.cfg_scale == False:
          sample_file = sample_file + '.png'
        else:
          sample_file = sample_file + f'_cfg_scale_{args.cfg_scale}.png'
        sample_image = skimage.io.imread(sample_file, as_gray=True)
        target_image = skimage.io.imread(target_file, as_gray=True)
        target_image = skimage.transform.resize(target_image, sample_image.shape)
        ssim = skimage.metrics.structural_similarity(
            sample_image, target_image, data_range=1.0, win_size=7
        )
        ssims.append(ssim)
    ssims = np.array(ssims)
    total_ssims.append(ssims)
    if args.save:
      metric_name = 'ssim'
      np.save(os.path.join(args.viz_folder, f'{metric_name}_seed_{gen_seed}.npy'), ssims)
    print(f"SSIM for seed {gen_seed}: {ssims.mean():.6f}")

  total_ssims = np.concatenate(total_ssims)
  mean_ssim = np.mean(total_ssims)
  print(f"Total SSIM is: {mean_ssim}")
  args["SSIM"] = mean_ssim

  now = datetime.now()
  timestamp_string = now.strftime("image_similarity_results_v2_%Y-%m-%d_%H:%M:%S.json")

  with open(os.path.join(args.viz_folder, timestamp_string), 'w') as f:
    json.dump(args, f, indent=2, sort_keys=True)


if __name__ == '__main__':
  cfg_scales = [3.0]
  args = EasyDict(
      {
        'save': True,
        'save_reps': True,
        'gen_seeds': [3, 5,7,11,13,17,42,53,83,1357], #[42, 53, 83, 1357], #,
        'num_samples': 100, #250, #100,
        'num_neighbours': 1,
        'cfg_scale': 3.0, #6.0, #False, 3.0, 6.0, 9.0
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_ood_repx2_spklray/scales/last/gamma1/val" #1.8, 0.8, 0.4
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_ood/last/gamma1/val" #2.0, 0.1, 0.2, 0.5
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2/last/gamma1/val" #75.1 ##72.4, 75.3, 75.4, 74.1
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_repx2_spklray/last/gamma1/val", #72.6, 75.0, 74.0, 72.8
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_repx2_spklray/last_3/gamma1/val", #
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_repx1_spklray/last_3/gamma1/val", #75.2 (3)
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_repx1_spklray_p0/last/gamma1/val", #75.2 (3)
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_repx1_spklray_p0.85/last/gamma1/val", #75.2 (3)
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_repx2_spklray/last_3/gamma1/val", #
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_repx2_spklray_p0.5/last/gamma1/val", # 
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_ood_repx2_mono/scales/last/gamma1/val", #0.4, 0.9, 0.4
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_ood_repx2_gauss/scales/last/gamma1/val", # ? , ?, 0.1

        #New
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_repx1_spklray_p0_6gpu/last/gamma_1/val",
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_repx1_spklray_p0_no_accum/last/gamma_1/val",
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_repx5_spklray_p0/last/gamma_1/val",
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_repx100_spklray_p0/last/gamma_1/val",
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_repx100_spklray_p0/last/gamma_1/val",
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_repx100_spklray_p0/last_4/gamma_1/val",

        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_ood_repx100_spklray_p0/last/gamma_1/val", #0.9
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_ood_t72_repx100_spklray_p0/last/gamma_1/val", #12.7
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_ood_t72_repx1_spklray_p0/last/gamma_1/val", #9.2
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_ood_repx1_spklray_p0_lapcutmix/last/gamma_1/val", #1.6 with spkl+mono, 0.6 with simclr
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_ood_t72_repx100_spklray_p0_lapcutmix/last/gamma_1/val", #18.3
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_ood_t72_repx2_spklray_p0_lapcutmix/last/gamma_1/val", #
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_ood_t72_repx5_spklray_p0_lapcutmix/last/gamma_1/val", #
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_ood_t72_repx1_spklray_p0_lapcutmix/last/gamma_1/val", #
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz2/mstar2_ood_t72/last/gamma_1/val", #

        # updated
        #'viz_folder': "/home/amirr/nfs_extra/data/zero123_logs_July2024/viz2/mstar2_ood_t72_repx1_spklray_p0/last/gamma_1/val"
        #'viz_folder': "/home/amirr/nfs_extra/data/zero123_logs_July2024/viz2/mstar2_ood_t72/last/gamma_1/val"
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2_repx1_spklray_p0/last/gamma1/val",
        #'viz_folder': "/home/amirr/codes/zero123/zero123/viz/mstar2/last/gamma1/val"

        #gl
        'viz_folder': "/home/amirr/codes/zero123/zero123/logs/viz/lambda_4/last/gamma_1/val"
      }
  )

  viz_folders = [
      #"/home/amirr/nfs_extra/data/zero123_logs_July2024/viz2/mstar2_ood_t72_repx1_spklray_p0/last/gamma_1/val",
      #"/home/amirr/nfs_extra/data/zero123_logs_July2024/viz2/mstar2_ood_t72/last/gamma_1/val",
      "/home/amirr/codes/zero123/zero123/logs/viz2/aff100x/last/gamma_1/val",
      "/home/amirr/codes/zero123/zero123/viz/mstar2_repx1_spklray_p0/last/gamma1/val",
      "/home/amirr/codes/zero123/zero123/viz/mstar2/last/gamma1/val",
      "/home/amirr/codes/zero123/zero123/logs/viz2/lambda_0.5/last/gamma_1/val",
  ]

  for viz_folder in viz_folders:
    for cfg_scale in cfg_scales:
      args.viz_folder = viz_folder
      args.cfg_scale = cfg_scale
      evaluate_ssim(args)


