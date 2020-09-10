from setuptools import setup, find_packages

__version__ = '0.1.2-alpha'
url = 'https://github.com/georgianpartners/Multimodal-Toolkit'

install_requires = [
    'torch',
    'transformers>=3.0',
    'numpy',
    'tqdm',
    'scipy',
    'networkx',
    'scikit-learn',
    'pandas',
]

setup(
    name='multimodal_transformers',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    license='MIT',
    description='Multimodal Extension Library for PyTorch HuggingFace Transformers',
    author='Ken Gu',
    author_email='kgu@georgianpartners.com',
    url=url,
    download_url='{}/archive/v_{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'multimodal', 'transformers', 'hugging-face'],   # Keywords that define your package best
    install_requires=install_requires,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
  ],
)