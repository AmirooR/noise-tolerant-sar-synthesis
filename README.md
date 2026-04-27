# Noise-tolerant novel-view SAR synthesis

This repository contains code for **Noise-Tolerant Novel-View SAR Synthesis via Denoising Diffusion** (Rahimi & Yu, *IEEE Transactions on Geoscience and Remote Sensing*, 2026). The implementation is based on [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) (Liu et al., ICCV 2023).

Training hyperparameters and experiment setups are defined as YAML configs under **`zero123/configs`** (including subfolders such as `zero123/configs/lambda/`).

## Representation learning

Contrastive representation learning code lives in **`representation_learning/`**. The main entry point is **`representation_learning/train_contrastive.py`**.

The **`model.py`** and **`layers.py`** modules in that folder are inspired by the official [Co-Domain Symmetry for Complex-Valued Deep Learning (CDS)](https://github.com/sutkarsh/cds) implementation (Singhal, Xing & Yu, CVPR 2022). This repository does not vendor that codebase; only the local files under `representation_learning/` implement the project’s training stack, with structure and layer ideas drawn from CDS.

## Citation

```bibtex
@article{rahimi2026noise,
  title={Noise-Tolerant Novel-View SAR Synthesis via Denoising Diffusion},
  author={Rahimi, Amir and Yu, Stella},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
```

For the underlying Zero-1-to-3 work, see the [original project](https://zero123.cs.columbia.edu/) and its paper.
