import os
from setuptools import setup, find_packages


# See: https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


__version__ = get_version("multimodal_transformers/__init__.py")
url = "https://github.com/georgianpartners/Multimodal-Toolkit"

install_requires = [
    "accelerate>=0.29.3",
    "importlib-metadata~=7.1.0",
    "networkx~=2.6.3",
    "numpy~=1.26.4",
    "pandas~=2.2.2",
    "pytest~=7.2.2",
    "sacremoses~=0.0.53",
    "scikit-learn~=1.5.1",
    "scipy~=1.13.0",
    "torch>=2.0.1",
    "transformers>=4.40.1",
    "tqdm",
]

setup(
    name="multimodal_transformers",
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    license="MIT",
    description="Multimodal Extension Library for PyTorch HuggingFace Transformers",
    author="Ken Gu",
    author_email="kgu@georgianpartners.com",
    maintainer="Akash Saravanan",
    maintainer_email="akash.saravanan@georgian.io",
    url=url,
    download_url="{}/archive/v_{}.tar.gz".format(url, __version__),
    keywords=[
        "pytorch",
        "multimodal",
        "transformers",
        "huggingface",
    ],  # Keywords that define your package best
    install_requires=install_requires,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
