from setuptools import setup

from torch.utils import cpp_extension

project_root = '/home/ubuntu/hh1001/780_project'


setup(
    name='fp8_gemm',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='fp8_gemm',
            sources=[
                f'{project_root}/csrc/gemm_wrapper.cpp',
                f'{project_root}/csrc/fp8_gemm_with_scale.cu',
                f'{project_root}/csrc/bf16_gemm.cu',
            ],
            include_dirs=[
                f'{project_root}/csrc',
                f'{project_root}/deps/cutlass/include',
                "/usr/local/cuda/include",
            ],
             library_dirs=[
                '/usr/lib/x86_64-linux-gnu',
            ],
            # library_dirs=[
            #     "/usr/local/cuda/lib64",
            #     "/usr/local/cuda/lib64/stubs"
            # ],
            libraries=["cudart", "cudadevrt", "cublas", "cudnn"],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '--expt-relaxed-constexpr',
                    '-Xcompiler=-fPIC',
                    f'-I{project_root}/csrc',
                    f'-I{project_root}/deps/cutlass',
                    '-arch=sm_90a',
                    '-gencode=arch=compute_90a,code=sm_90a',  # Hopper
                ]
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)