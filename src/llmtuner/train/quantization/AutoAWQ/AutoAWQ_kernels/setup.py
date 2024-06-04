import os
import torch
from pathlib import Path
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_lib
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"
AUTOAWQ_KERNELS_VERSION = "0.0.6"
PYPI_BUILD = os.getenv("PYPI_BUILD", "0") == "1"
CUDA_VERSION = os.getenv("CUDA_VERSION", None) or torch.version.cuda
ROCM_VERSION = os.environ.get("ROCM_VERSION", None) or torch.version.hip


if not PYPI_BUILD:
    # only adding CUDA/ROCM version if we are not building for PyPI to comply with PEP 440
    if CUDA_VERSION:
        CUDA_VERSION = "".join(CUDA_VERSION.split("."))[:3]
        AUTOAWQ_KERNELS_VERSION += f"+cu{CUDA_VERSION}"
    elif ROCM_VERSION:
        ROCM_VERSION = "".join(ROCM_VERSION.split("."))[:3]
        AUTOAWQ_KERNELS_VERSION += f"+rocm{ROCM_VERSION}"
    else:
        raise RuntimeError(
            "Your system must have either Nvidia or AMD GPU to build this package."
        )

print(f"Building AutoAWQ Kernels version {AUTOAWQ_KERNELS_VERSION}")

common_setup_kwargs = {
    "version": AUTOAWQ_KERNELS_VERSION,
    "name": "autoawq_kernels",
    "author": "Casper Hansen",
    "license": "MIT",
    "python_requires": ">=3.8.0",
    "description": "AutoAWQ Kernels implements the AWQ kernels.",
    "long_description": (Path(__file__).parent / "README.md").read_text(
        encoding="UTF-8"
    ),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/casper-hansen/AutoAWQ_kernels",
    "keywords": ["awq", "autoawq", "quantization", "transformers"],
    "platforms": ["linux", "windows"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
}

requirements = [
    "torch>=2.0.1",
]


def get_include_dirs():
    include_dirs = []

    if CUDA_VERSION:
        conda_cuda_include_dir = os.path.join(
            get_python_lib(), "nvidia/cuda_runtime/include"
        )
        if os.path.isdir(conda_cuda_include_dir):
            include_dirs.append(conda_cuda_include_dir)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    include_dirs.append(this_dir)

    return include_dirs


def get_generator_flag():
    generator_flag = []

    # if CUDA_VERSION:
    torch_dir = torch.__path__[0]
    if os.path.exists(
        os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")
    ):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    return generator_flag


def get_compute_capabilities(
    compute_capabilities={75, 80, 86, 89, 90}
):
    capability_flags = []

    if CUDA_VERSION:
        # Collect the compute capabilities of all available CUDA GPUs
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            cc = major * 10 + minor
            if cc < 75:
                raise RuntimeError(
                    "GPUs with compute capability less than 7.5 are not supported."
                )

        # Figure out compute capability
        for cap in compute_capabilities:
            capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

    return capability_flags


def get_extra_compile_args(arch_flags, generator_flags):
    extra_compile_args = {}

    if os.name == "nt" and CUDA_VERSION:
        include_arch = os.getenv("INCLUDE_ARCH", "1") == "1"
        # Relaxed args on Windows
        if include_arch:
            extra_compile_args = {"nvcc": arch_flags}

    elif CUDA_VERSION:
        extra_compile_args = {
            "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-DENABLE_BF16",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ]
            + arch_flags
            + generator_flags,
        }

    return extra_compile_args


def get_extra_link_args():
    extra_link_args = []

    if os.name == "nt" and CUDA_VERSION:
        cuda_path = os.environ.get("CUDA_PATH", None)
        extra_link_args = ["-L", f"{cuda_path}/lib/x64/cublas.lib"]

    return extra_link_args


include_dirs = get_include_dirs()
extra_link_args = get_extra_link_args()
generator_flags = get_generator_flag()
arch_flags = get_compute_capabilities()
extra_compile_args = get_extra_compile_args(arch_flags, generator_flags)


extensions = []
if CUDA_VERSION:
    # contain un-hipifiable inline PTX
    extensions.append(
        CUDAExtension(
            "awq_ext",
            [
                "awq_ext/pybind_awq.cpp",
                "awq_ext/quantization/gemm_cuda_gen.cu",
                "awq_ext/layernorm/layernorm.cu",
                "awq_ext/position_embedding/pos_encoding_kernels.cu",
                "awq_ext/quantization/gemv_cuda.cu",
                "awq_ext/vllm/moe_alig_block.cu",
                "awq_ext/vllm/activation.cu",
                "awq_ext/vllm/topk_softmax_kernels.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    )

    # only compatible with ampere
    arch_flags = get_compute_capabilities({80, 86, 89, 90})
    extra_compile_args_v2 = get_extra_compile_args(arch_flags, generator_flags)

    extensions.append(
        CUDAExtension(
            "awq_v2_ext",
            [
                "awq_ext/pybind_awq_v2.cpp",
                "awq_ext/quantization_new/gemv/gemv_cuda.cu",
                "awq_ext/quantization_new/gemm/gemm_cuda.cu",
            ],
            extra_compile_args=extra_compile_args_v2,
        )
    )

extensions.append(
    CUDAExtension(
        "exl_ext",
        [
            "awq_ext/exllama/exllama_ext.cpp",
            "awq_ext/exllama/cuda_buffers.cu",
            "awq_ext/exllama/cuda_func/column_remap.cu",
            "awq_ext/exllama/cuda_func/q4_matmul.cu",
            "awq_ext/exllama/cuda_func/q4_matrix.cu",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
)
extensions.append(
    CUDAExtension(
        "exlv2_ext",
        [
            "awq_ext/exllamav2/ext.cpp",
            "awq_ext/exllamav2/cuda/q_matrix.cu",
            "awq_ext/exllamav2/cuda/q_gemm.cu",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
)

if os.name != "nt" and CUDA_VERSION:
    # FasterTransformer kernels
    extensions.append(
        CUDAExtension(
            "awq_ft_ext",
            [
                "awq_ext/pybind_awq_ft.cpp",
                "awq_ext/attention/ft_attention.cpp",
                "awq_ext/attention/decoder_masked_multihead_attention.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    )

additional_setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": {"build_ext": BuildExtension},
}

common_setup_kwargs.update(additional_setup_kwargs)

setup(
    packages=find_packages(),
    install_requires=requirements,
    include_dirs=include_dirs,
    **common_setup_kwargs,
)
