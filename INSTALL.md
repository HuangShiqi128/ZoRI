## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd zori/modeling/pixel_decoder/ops
sh make.sh
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Example conda environment setup
```bash
conda create --name zori python=3.8 -y
conda activate zori
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
git clone https://github.com/HuangShiqi128/ZoRI
cd zori
pip install -r requirements.txt
cd zori/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
pip install -U openmim
mim install mmcv
```
