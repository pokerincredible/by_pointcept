import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

# 若系统 nvcc 为 CUDA 12.4（不支持 sm_120），可设 TORCH_CUDA_ARCH_LIST=9.0 再编译，
# 生成的 kernel 在 RTX 5090 上靠前向兼容运行
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

setup(
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops"],
    package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
