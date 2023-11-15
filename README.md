# pySynDNA

A python implementation and extension of synDNA ("a Synthetic DNA Spike-in Method for Absolute Quantification of Shotgun Metagenomic Sequencing", Zaramela et al., mSystems 2022)

## Installation

To install this notebook, first clone the repository from GitHub:

```
git clone https://github.com/biocore/pysyndna.git
```

Create a Python3 Conda environment in which to run the notebook:

```
conda create -n pysyndna 'python=3.9' flake8 nose numpy pandas pep8 pyyaml scikit-learn scipy
```

Activate the Conda environment:

```
conda activate pysyndna
```

Install the `biom-format` package from the `conda-forge` channel:

```
conda install -c conda-forge biom-format
```

Change directory to the downloaded repository folder and install:

```
cd pysyndna
pip install -e .
```
