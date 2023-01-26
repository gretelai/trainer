import pathlib
from setuptools import setup, find_packages

local_path = pathlib.Path(__file__).parent
install_requires = (local_path / "requirements.txt").read_text().splitlines()

mysql_extras = [
    "mysqlclient",
]
postgres_extras = [
    "psycopg2-binary",
]
snowflake_extras = [
    "snowflake-sqlalchemy",
]
connectors_extras = mysql_extras + postgres_extras + snowflake_extras

setup(name="gretel-trainer",
      use_scm_version=True,
      setup_requires=["setuptools_scm"],
      package_dir={'': 'src'},
      install_requires=install_requires,
      extras_require={
          "connectors": connectors_extras,
          "mysql": mysql_extras,
          "postgres": postgres_extras,
          "snowflake": snowflake_extras,
      },
      python_requires=">=3.7",
      packages=find_packages("src"),
      package_data={'': ['*.yaml']},
      include_package_data=True,
      description="Synthetic Data Generation with optional Differential Privacy",
      url="https://github.com/gretelai/gretel-trainer",
      license="https://gretel.ai/license/source-available-license",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free To Use But Restricted",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ]
)
