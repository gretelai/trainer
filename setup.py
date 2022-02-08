import pathlib
from setuptools import setup, find_packages

local_path = pathlib.Path(__file__).parent
install_requires = (local_path / "requirements.txt").read_text().splitlines()

setup(name="trainer", package_dir={'': 'src'}, install_requires=install_requires, packages=find_packages("src"))
