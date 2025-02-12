from setuptools import setup, find_packages

setup(
    name='parglm_torch',
    version='0.1.0',
    description='Multivariate simulation and GLM analysis with PyTorch',
    author='Michael Sorochan Armstrong',
    author_email='mdarmstr@ugr.es',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scipy',
        'seaborn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
