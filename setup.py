from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

version = {}
with open(path.join(here, "prepUS", "__version__.py")) as f:
    exec(f.read(), version)

setup(
    name="prepUS",
    version=version["__version__"],
    description="Utily script for ultrasound videos pre-processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chop1/prepUS",
    author="MEYER Adrien",
    license="Apache Software License 2.0",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="ultrasound preprocessing",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["sonocrop", "mmcv-lite"],
    entry_points={  # Optional
        "console_scripts": [
            "prepUS=prepUS.cli:main",
        ],
    },
)
