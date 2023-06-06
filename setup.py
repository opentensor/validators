# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import pathlib
import pkg_resources
from setuptools import setup


def read(fname):
    this_directory = pathlib.Path(__file__).parent
    long_description = (this_directory / fname).read_text()
    return long_description


def read_requirements(path):
    with pathlib.Path(path).open() as requirements_txt:
        return [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


requirements = read_requirements("requirements.txt")


setup(
    name="openvalidators",
    version=get_version("openvalidators/__init__.py"),
    description="Openvalidators is a collection of open source validators for the Bittensor Network.",
    url="https://github.com/opentensor/validators",
    author="bittensor.com",
    packages=["openvalidators"],
    include_package_data=True,
    author_email="",
    license="MIT",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": ["foundation-validator = openvalidators.neuron:main"],
    },
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    maintainer="",
    maintainer_email="",
    keywords=[
        "bittensor",
        "validator",
        "ai",
        "machine-learning",
        "deep-learning",
        "blockchain",
        "pytorch",
        "torch",
        "neural-networks",
        "cryptocurrency",
    ],
)
