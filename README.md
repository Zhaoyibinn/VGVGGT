<div align="center">

# $\text{VG}^2\text{GT}$: Voxel-Gaussian Splatting Visual Geometry Grounded Transformer

<!-- <a href="https://jytime.github.io/data/VGGT_CVPR25.pdf" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a> -->
<a href="https://arxiv.org/abs/2606.01573"><img src="https://img.shields.io/badge/arXiv-2606.01573-red?logo=arxiv" alt="arXiv"></a>
<!-- <a href="https://vgg-t.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/spaces/facebook/vggt'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> -->


**[3D Vision Group, East China University of Science and Technology](https://www.ecust.edu.cn/)**; 


[Yibin Zhao](https://github.com/Zhaoyibinn)
</div>

```bibtex
@misc{zhao2026textvg2gtvoxelgaussiansplattingvisual,
      title={$\text{VG}^2$GT: Voxel-Gaussian Splatting Visual Geometry Grounded Transformer}, 
      author={Yibin Zhao and Yihan Pan and Jun Nan and Wenli Yang and Liwei Chen and Jianjun Yi},
      year={2026},
      eprint={2606.01573},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.01573}, }
```



## Overview

$\text{VG}^2\text{GT}$ (Voxel-Gaussian Splatting Visual Geometry Grounded Transformer) is a feed-forward, **non-pixel-aligned** Gaussian splatting reconstruction method that maps images to Gaussian splatting scenes within seconds, demonstrating significant advantages in novel view synthesis (NVS), depth map estimation, and surface reconstruction, while maintaining efficient training and inference costs.

## Environment Setup

This code has been tested with Torch 2.5.0 + CUDA 11.8 and Torch 2.7.1 + CUDA 12.8.

First, clone this repository to your local machine, and install the dependencies. 

```bash
git clone https://github.com/Zhaoyibinn/VGVGGT.git
cd VGVGGT
conda env create -f environment.yaml
```

Install the differentiable renderer based on stochastic volumetric rendering
```bash
git clone --recursive https://github.com/Zhaoyibinn/Geometry-Grounded-Gaussian-Splatting.git
cd Geometry-Grounded-Gaussian-Splatting
pip install . --no-build-isolation
```

(Optional) Install the original 3DGS rasterization renderer (with depth map rendering)
```bash
git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
git checkout 9c5c202
pip install . --no-build-isolation
```

(Optional) Install the 2DGS rasterization renderer
```bash
git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git
cd 2d-gaussian-splatting
pip install . --no-build-isolation
```

## Data Download
### Training Data Download
Taking the IGGT infinigen dataset as an example, all training data should be placed under the `train_data` folder. The default data structure is `train_data/iggt/processed_infinigen/extracted/******`, where each scene is stored in its own subdirectory.

<a href="https://huggingface.co/datasets/lifuguan/InsScene-15K"><img src="https://img.shields.io/badge/Data-IGGT-blue?logo=huggingface" alt="Data IGGT"></a>

### Pre-trained Weights Download
We use DA3_Giant_1.1 as the default pre-trained weights. All pre-trained weights should be placed under the `weight` folder. The default structure is `weight/da3/weights_gaint_1.1/model.safetensors`.

<a href="https://huggingface.co/depth-anything/DA3-GIANT-1.1"><img src="https://img.shields.io/badge/PreTrained-IGGT-blue?logo=huggingface" alt="PreTrained IGGT"></a>

## Model Training

Our model training framework is based on VGGT, with modifications made on top of it.

The model is primarily located under `vggt/models`. By default, we train and perform inference using `vggt/models/depthanything3.py`.


The default training configuration file is located at `training/config/demo.yaml`.

You can start training directly:
```bash
source training/launch.sh
python training/launch.py --config demo
```

If you have multiple GPUs, you can also use multi-GPU training to improve efficiency:
```bash
bash training/launch2.sh
```

The training results will be saved under `logs/demo`, where a copy of the YAML configuration will also be saved by default for inference.

### Configuration File Description
In the configuration file, you can adjust the following key parameters:
1. `gs_mode`: Controls the type of Gaussian splatting renderer. Available options: `2DGS`, `3DGS`, `GGGS`
2. `img_size`, `aspects`: Control the long-edge resolution and short/long edge ratio of training images respectively. If you keep encountering OOM errors, consider lowering these values
3. `img_nums`: The range of the number of images sampled during training. Similarly, lower this range if you encounter OOM issues
4. `low_resolution_voxelsize`: The voxel size used for Gaussian scene decoding
5. `frozen_module_names`: Modules to freeze during training. Only `model.backend` and `*backend_gs_decoder*` need to be trained

## Model Inference
We also provide Colmap-based model inference and saving, along with NVS performance evaluation and depth map synthesis evaluation (note: for datasets that require masks, such as DTU and T&T, you may still need to recompute the metrics).

The input data folder only needs to contain an `images` folder.
```bash
python demo_colmap.py --scene_dir <data_folder> --shared_camera --conf_thres_value 0.0 --config_file <config_yaml_path> --checkpoint_path <checkpoint_path> --sparse_subdir <colmap_output_folder>
```

## Acknowledgements

Thanks to these great repositories: [VGGT](https://github.com/facebookresearch/vggt), [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3), [amb3r](https://github.com/HengyiWang/amb3r), [IGGT](https://github.com/lifuguan/IGGT_official), [GGGS](https://baowenz.github.io/geometry_grounded_gaussian_splatting/), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [2DGS](https://github.com/hbb1/2d-gaussian-splatting/) and many other inspiring works in the community.
