import os
import sysconfig
import subprocess

from pathlib import Path


def build_shared(src_files, target, mode="release"):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
    
    src_files = [str(p) for p in src_files]
    target = str(target)

    # CUDA
    cuda_home = (
        os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    )
    cuda_inc = os.path.join(cuda_home, "include")
    cuda_lib = os.path.join(cuda_home, "lib64")
    if not os.path.isdir(cuda_inc) or not os.path.isdir(cuda_lib):
        raise SystemExit(f"CUDA not found. Set CUDA_HOME or install to {cuda_home}")

    # Python
    py_inc = sysconfig.get_paths()["include"]

    # Torch
    t_inc = include_paths()
    t_lib = library_paths()

    cxx11_abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1)
    abi_macro = f"-D_GLIBCXX_USE_CXX11_ABI={int(cxx11_abi)}"
    # target_name = target.split(".")[0]
    target_name = os.path.splitext(os.path.basename(target))[0]

    print(">> NVCC Compile...")
    cmd = [
        "nvcc",
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        "-I" + py_inc,
        "-I" + cuda_inc,
        *[f"-I{p}" for p in t_inc],
        # ABI macro
        abi_macro,
        f"-DTORCH_EXTENSION_NAME={target_name}",
        # libs
        "-L" + cuda_lib,
        *[f"-L{p}" for p in t_lib],
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        cuda_lib,
        *[arg for p in t_lib for arg in ("-Xlinker", "-rpath", "-Xlinker", p)],
        # link against torch and CUDA runtime
        "-lc10",
        "-lc10_cuda",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-ltorch",
        "-ltorch_python",
        "-lcudart",
    ]
    if mode == "release":
        cmd.append("-O3")
    else:
        cmd.extend(["-g", "-G", "-lineinfo", "-DTORCH_USE_CUDA_DSA"])

    assert isinstance(src_files, list) or isinstance(src_files, tuple)
    cmd.extend(src_files)
    cmd.extend(["-o", target])
    print(">> Building so with:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":

    build_path = Path(__file__).resolve().parent
    src_path = build_path.parent / "src"
    so_path = build_path / "esa_kernel.so"

    cu_files = list(src_path.glob("*.cu"))
    cc_files = list(src_path.glob("*.cc"))

    build_shared(cu_files + cc_files, so_path)
