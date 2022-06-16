import pathlib
from setuptools import setup, find_packages

local_path = pathlib.Path(__file__).parent
install_requires = (local_path / "requirements.txt").read_text().splitlines()

setup(name="gretel-trainer",
      version="0.2.1",
      package_dir={'': 'src'}, 
      install_requires=install_requires, 
      python_requires=">=3.7",
      packages=find_packages("src"),
      package_data={'': ['*.yaml']},
      include_package_data=True,
      description="Synthetic Data Generation with optional Differential Privacy",
      url="https://github.com/gretelai/gretel-trainer",
      license="http://www.apache.org/licenses/LICENSE-2.0",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ]
)
