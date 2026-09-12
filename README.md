# HybricMamba

Project for image classification with hybrid CNN-Mamba models and standard CNN/ViT baselines. The main training pipeline is implemented in `Train.py`.

## 1. Model architecture

### HybricMamba

`HybricMamba` is defined in `models/CNN_Mamba_CNN_Mamba_Enhanced/HybricMamba.py`. It combines convolutional stages with 2D selective-scan Mamba stages:

```text
Input image (3 x 32 x 32)
        |
        v
Stem: Conv2d -> BatchNorm -> SiLU -> Conv2d -> SimAM
        |
        v
Stage 1: CNNStage / MBConvSimAM + downsampling
        |
        v
Stage 2: MambaStage / MambaPatchMerging + TriBranchMambaBlockV4
        |
        v
Stage 3: CNNStage / MBConvSimAM + downsampling
        |
        v
Stage 4: MambaStage / MambaPatchMerging + TriBranchMambaBlockV4
        |
        v
Multi-stage dual pooling: average pooling + max pooling
        |
        v
LayerNorm -> ECAAttention1D -> Dropout -> Linear classifier
```

The Mamba blocks split features into three branches:

- **SSM branch:** processes spatial information with the 2D selective-scan operator.
- **Convolution branch:** uses local 3x3 and 5x5 convolutions.
- **Identity branch:** preserves part of the input representation.

During training, when `use_aux=True`, the model returns three outputs:

```python
(main_logits, aux_stage2_logits, aux_stage3_logits)
```

`Train.py` computes the main classification loss and adds `0.3 * CrossEntropyLoss` for each auxiliary output. During evaluation (`model.eval()`), the model returns only `main_logits`.

### Model variants

`Train.py` provides these model names:

| Name | Description |
|---|---|
| `LIGHT_HYBRIC_MAMBA` | Small HybricMamba configuration |
| `MEDIUM_HYBRIC_MAMBA` | Medium HybricMamba configuration |
| `HEAVY_HYBRIC_MAMBA` | Larger HybricMamba configuration |
| `SUPER_MAMBA_DEPT_3` | Super Mamba with depth 3 |
| `SUPER_MAMBA_DEPT_4` | Super Mamba with depth 4 |
| `VGG16` | timm VGG16 baseline |
| `RESNET18` | timm ResNet18 baseline |
| `VIT_B` | ViT-Base, configured for 32x32 input |
| `VIT_S` | ViT-Small, configured for 32x32 input |
| `EFFICIENTNET_B0` | EfficientNet-B0 baseline |
| `MOBILENETV3_SMALL` | MobileNetV3-Small baseline |
| `GHOSTNET` | GhostNet baseline |

The exact constructor parameters are defined in `build_Model()` inside `Train.py`. The same parameters must be used when loading a checkpoint for evaluation or benchmarking.

## 2. Project structure

```text
Train.py                         Main training and test-evaluation pipeline
model_metric.py                  Model benchmark: accuracy, FLOPs, latency, memory
Dataloader/DATASET.py            Dataset scanner and PyTorch Dataset
models/CNN_Mamba_.../HybricMamba.py
                                  HybricMamba implementation
models/vmamba/Vmamba_ultils.py   Super_Mamba implementation
models/vmamba/VSSBlock_ultils.py  Selective-scan and VSS operations
data_split_utils.py              Persistent train/validation/test split helper
Predict_on_Full_dataset.py       Prediction over a complete ImageFolder dataset
predict.py                       Prediction for selected individual images
eval.py                          Older standalone evaluation script
benchmark_outputs/               Saved benchmark CSV files
Ressult/TFJ/                     Checkpoints, logs and confusion matrices
kernels/selective_scan/          Optional CUDA selective-scan extension
```

## 3. Environment requirements

Recommended environment:

- Linux is recommended for compiling and running the CUDA selective-scan extension.
- Python 3.10 or newer.
- NVIDIA GPU and a CUDA toolkit compatible with the installed PyTorch version.
- PyTorch, torchvision and torchaudio.
- CUDA compiler (`nvcc`), `ninja` and a C++ compiler.

The repository does not currently provide a root-level `requirements.txt`. The selective-scan package lists these main dependencies:
- `torch==2.13.0`
- `torchvision==0.28.0`
- `torchaudio==2.11.0`
- `triton==3.7.1`
- `ninja==1.13.2`
- `einops==0.8.2`
- `packaging==26.3`
- `timm==0.4.12`
- `fvcore==0.1.5.post20221221`
- `pytest==9.1.1`
- `chardet==7.6.0`
- `yacs==0.1.8`
- `termcolor==3.3.0`
- `submitit==1.5.4`
- `tensorboardX==2.6.5`
- `tensorboard`
- `seaborn==0.13.2`
- `matplotlib==3.10.9`
- `numpy==2.2.6`
- `pandas==2.3.3`
- `scipy==1.15.3`
- `scikit-learn==1.7.2`
- `pillow==12.3.0`
- `opencv-python==5.0.0.93`
- `tqdm==4.70.0`
- `thop==0.1.1-2209072238`
- `PyYAML==6.0.3`


## 4. Installation

Step 1: Clone the HybricMamba repository

```bash
git clone https://github.com/Bill-Ds-lab/HybircMamba.git
cd HybricMamba

```

Step 2: Create the Conda environment
The project uses Python 3.10 and is recommended to run in a Conda environment.
If `environment.yml` is available, recreate the tested environment directly:

```bash
conda env create -f environment.yml
conda activate HybricMamba

```

Alternatively, create a new environment manually:

Step 1

```bash

conda create -n HybricMamba python=3.10.20 pip -y
conda activate HybricMamba
conda install -c nvidia cuda-toolkit=13.0


```

Step 2: Install Python dependencies
Install the main Python dependencies:

```bash
pip install -r requirements.txt
cd kernels/selective_scan
pip install --no-build-isolation .


```




Check PyTorch and CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

```

Verify the selective-scan extension:

```bash
python -c "import selective_scan_cuda_core; print('selective_scan_cuda_core: OK')"

```

A successful installation should report that PyTorch can access CUDA and that the selective-scan CUDA extension can be imported.

Step 6: Run the project
After the environment has been installed and the dataset has been prepared, the main training pipeline can be started with:

```bash
python Train.py

```

For model benchmarking:

```bash
python model_metric.py

```

The exact model, dataset, and training parameters are configured in the corresponding scripts.

If compilation fails, check that `nvcc`, `CUDA_HOME`, PyTorch CUDA, and the GPU compute capability are compatible. The Mamba layers in this project are not intended to be a CPU-only training path.
