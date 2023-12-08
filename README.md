# pySynDNA

A python implementation and extension of SynDNA 
("a Synthetic DNA Spike-in Method for Absolute Quantification of Shotgun 
Metagenomic Sequencing", Zaramela et al., mSystems 2022)

## Overview

The package provides an API for generating absolute cell count estimates from
metagenomic sequencing data that includes synthetic DNA (synDNA) spike-ins.
The general approach, as outlined in Zaramela et al., is to use counts of 
specific synDNAs present in known masses to fit a linear regression model for 
each sample that predicts the gDNA mass of a sequence based on its counts in 
that sample.  These linear models are subsequently applied to the counts of 
microbial genomes in each sample to calculate the mass of each such genome, 
and those masses are then transformed into the number of instances of each 
genome (that is, the approximate cell count of that microbe) in the input gDNA. 
pySynDNA extends this approach by also calculating the approximate cell count
of each microbe in the input sample material (before gDNA extraction).

## Installation

To install this package, first clone the repository from GitHub:

```
git clone https://github.com/AmandaBirmingham/pysyndna.git
```

Change directory into the new `pysndna` folder and create a 
Python3 Conda environment in which to run the software:

```
conda env create -n pysyndna -f environment.yml  
```

Activate the Conda environment and install the package:

```
conda activate pysyndna
pip install -e .
```

## General Usage

First, calculate linear regression models per sample by calling the 
`fit_linear_regression_models` function in the `src.fit_syndna_models` module 
(or, for qiita-based usage, the `fit_linear_regression_models_for_qiita` 
function).

Second, apply these models to counts of microbial genomes in the 
samples by calling the `calc_ogu_cell_counts_biom` function in the 
`src.calc_cell_counts` module, specifying whether to return the cell counts 
per gram of gDNA or the cell counts per gram of un-extracted sample material 
(the latter is usually the more relevant metric).  For qiita-based usage, 
call the `calc_ogu_cell_counts_per_g_of_sample_for_qiita` function (which 
always returns the cell counts per gram of un-extracted sample material.)

The documentation for these functions offer details on the specifics of 
their parameters, and the unit tests provide functional examples of inputs 
and outputs.