# AutoAWQ Kernels

AutoAWQ Kernels is a new package that is split up from the [main repository](https://github.com/casper-hansen/AutoAWQ) in order to avoid compilation times.

## Requirements

- Windows: Must use WSL2.

- NVIDIA:
  - GPU: Must be compute capability 7.5 or higher.
  - CUDA Toolkit: Must be 11.8 or higher.
- AMD:
  - ROCm: Must be 5.6 or higher.

## Install

### Install from PyPi

The package is available on PyPi with CUDA 12.1.1 wheels:

```
pip install autoawq-kernels
```

### Install release wheels

For ROCm and other CUDA versions, you can use the wheels published at each [release](https://github.com/casper-hansen/AutoAWQ_kernels/releases/):

```
pip install https://github.com/casper-hansen/AutoAWQ_kernels/releases/download/v0.0.2/autoawq_kernels-0.0.2+rocm561-cp310-cp310-linux_x86_64.whl
```

### Build from source
You can also build from source:

```
git clone https://github.com/casper-hansen/AutoAWQ_kernels
cd AutoAWQ_kernels
pip install -e .
```

To build for ROCm, you need to first install the following packages `rocsparse-dev hipsparse-dev rocthrust-dev rocblas-dev hipblas-dev`.