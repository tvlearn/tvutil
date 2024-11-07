# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# install: pip install .
# develop: pip install --editable . // pip install -e .
import toml
import pathlib
import sysconfig
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile
from setuptools import setup, find_packages


pyproject_text = pathlib.Path("pyproject.toml").read_text()
pyproject_data = toml.loads(pyproject_text)
build_type = pyproject_data["build-system"]["build-type"]


BUILD_TYPES = {
    "Release": ["-O3", "-DNDEBUG"],
    "Debug": ["-O0", "-g"],
    "RelWithDebInfo": ["-O2", "-g", "-DNDEBUG"],
    "MinSizeRel": ["-Os", "-DNDEBUG"],
}

include_dirs = [
    "tvutil/prepost/utils/extern/eigen",
    "tvutil/prepost/utils/cpp/include",
]

extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
extra_compile_args += [
    "-Wall",
    "-Wextra",
    "-Wshadow",
    "-pedantic",
    "-Wno-unknown-pragmas",
    "-march=native",
]
extra_compile_args += BUILD_TYPES.get(build_type, [])

define_macros = [("TVUTIL_PRECISION", "double"), ("EIGEN_DONT_PARALLELIZE", None)]

ext_modules = [
    Pybind11Extension(
        "cppUtils",
        [
            "tvutil/prepost/utils/cpp/src/Bindings.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ["-fopenmp"],
        extra_link_args=["-lgomp"],
        define_macros=define_macros,
        language="c++",
        cxx_std=17,
    ),
]

with ParallelCompile(default=0):
    print(extra_compile_args, flush=True)
    setup(
        name="tvutil",
        version="0.1",
        packages=find_packages(exclude=("test",)),
        zip_safe=False,
        ext_modules=ext_modules,
        install_requires=[
            "numpy",
            "scikit-learn",
            "pandas",
            "h5py",
            "scipy",
        ],
    )
