import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup

setup(
    name="DFA3D_V2",
    version="0.1",
    description="A package for 3D face alignment using deep learning",
    author="Jianzhu Guo",
    url="https://github.com/cleardusk/3DDFA",
    packages=find_packages(exclude=["docs"]),
    license="MIT License",
    install_requires=["torch>=1.0", "torchvision>=0.2", "opencv-python>=4.1"],
    ext_modules=cythonize("DFA3D_V2/FaceBoxes/utils/nms/cpu_nms.pyx"),
    include_dirs=[numpy.get_include()],
    extras_require={
        "dev": [
            "flake8",
            "pytest",
        ]
    },
    package_data={"": ["weights/*", "configs/*"]},
)
