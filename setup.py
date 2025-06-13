"""
Setup script for building C++ extensions

This compiles our performance-critical C++ code into Python modules.
"""

from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Define C++ extensions
ext_modules = [
    Pybind11Extension(
        "tradeknowledge_cpp",
        sources=[
            "src/cpp/text_search.cpp",
            "src/cpp/similarity.cpp",
            "src/cpp/tokenizer.cpp",
            "src/cpp/bindings.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            "src/cpp/include"
        ],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        define_macros=[("VERSION_INFO", "1.0.0")],
    ),
]

setup(
    name="tradeknowledge",
    version="1.0.0",
    author="TradeKnowledge Team",
    description="High-performance book knowledge system for algorithmic trading",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "pybind11>=2.11.0",
        "numpy>=1.24.0"
    ],
    zip_safe=False,
)