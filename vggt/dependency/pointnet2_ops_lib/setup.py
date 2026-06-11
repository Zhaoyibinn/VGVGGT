import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2_ops", "_version.py")).read())

def _default_cuda_arch_list():
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return None

    try:
        import torch

        if not torch.cuda.is_available():
            return None

        capabilities = {
            f"{major}.{minor}"
            for major, minor in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))
        }
        if not capabilities:
            return None

        arch_list = sorted(capabilities, key=lambda value: tuple(int(part) for part in value.split(".")))
        arch_list[-1] = f"{arch_list[-1]}+PTX"
        return ";".join(arch_list)
    except Exception:
        return None


cuda_arch_list = _default_cuda_arch_list()
if cuda_arch_list:
    os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list

setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
