from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="hnsw-cpp",
    version="0.1.0",
    description="HNSW index in C++23 with NEON kernels, from Python",
    ext_modules=[
        Pybind11Extension(
            "hnsw_cpp",
            ["bindings.cpp"],
            include_dirs=["../include"],
            cxx_std=23,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
