#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=lambda_0.5
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G 
#SBATCH --time=48:00:00
#SBATCH --account=stellayu
#SBATCH --partition=stellayu
#SBATCH --gpus=2
#SBATCH -o ./slurm_outs/%x.out

# The application(s) to execute along with its input arguments and options:
#srun -n 1 -t 10:00:00 --cpus-per-task=16 --mem=64G --partition=stellayu --gpus=2 --pty bash
/bin/hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

. ~/.bashrc
conda activate zero123


## LAMBDA
NCCL_P2P_DISABLE=1 python main.py -t --base configs/lambda/MSTAR2_repx0.5_spklray_p0.yaml --gpus 0,1 --scale_lr False --num_nodes 1 --seed 42 --check_val_every_n_epoch 100 --finetune_from 105000.ckpt --name lambda_MSTAR2_repx0.5_p0

#NCCL_P2P_DISABLE=1 python main.py -t --base configs/lambda/MSTAR2_repx1_spklray_p0.yaml --gpus 0,1 --scale_lr False --num_nodes 1 --seed 42 --check_val_every_n_epoch 100 --finetune_from 105000.ckpt --name lambda_MSTAR2_repx1_p0

#NCCL_P2P_DISABLE=1 python main.py -t --base configs/lambda/MSTAR2_repx2_spklray_p0.yaml --gpus 0,1 --scale_lr False --num_nodes 1 --seed 42 --check_val_every_n_epoch 100 --finetune_from 105000.ckpt --name lambda_MSTAR2_repx2_p0

#NCCL_P2P_DISABLE=1 python main.py -t --base configs/lambda/MSTAR2_repx4_spklray_p0.yaml --gpus 0,1 --scale_lr False --num_nodes 1 --seed 42 --check_val_every_n_epoch 100 --finetune_from 105000.ckpt --name lambda_MSTAR2_repx4_p0

#NCCL_P2P_DISABLE=1 python main.py -t --base configs/lambda/MSTAR2_repx10_spklray_p0.yaml --gpus 0,1 --scale_lr False --num_nodes 1 --seed 42 --check_val_every_n_epoch 100 --finetune_from 105000.ckpt --name lambda_MSTAR2_repx10_p0

#NCCL_P2P_DISABLE=1 python main.py -t --base configs/lambda/MSTAR2_repx50_spklray_p0.yaml --gpus 0,1 --scale_lr False --num_nodes 1 --seed 42 --check_val_every_n_epoch 100 --finetune_from 105000.ckpt --name lambda_MSTAR2_repx50_p0
