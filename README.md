# pySynDNA

A python implementation and extension of SynDNA 
("a Synthetic DNA Spike-in Method for Absolute Quantification of Shotgun 
Metagenomic Sequencing", Zaramela et al., mSystems 2022)

## Overview

The package provides an API for generating absolute cell count estimates from
metagenomic shotgun sequencing data that includes synthetic DNA (synDNA) spike-ins by 
implementing and extending the technique of [Zaramela et al](https://pubmed.ncbi.nlm.nih.gov/36317886/). 
The calculation is performed in two parts: first, the known 
spiked-in mass of pooled synDNAs and the known concentrations of each synDNA 
within the pool are used to calculate the mass of each synDNA sequenced from a given sample. 
These mass values are paired with the sequenced counts of each synDNA in the sample, and 
a regression model is fitted to predict mass from counts within that sample 
as shown in the figure below.

![pySynDNA regression fit workflow](https://raw.githubusercontent.com/biocore/pysyndna/main/docs/absolute_quant_fit_models_workflow.png?raw=true)

The counts for each microbial genome in the sample are then translated into 
masses via the regression model, and the masses are converted to genome counts 
using genome lengths and Avogadro's number 
(see Zaramela et al [equation 2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9765022/#FD2)).
Assuming approximately one genome per cell, the calculated genome counts are
treated as approximate cell counts.  pySynDNA extends this approach further by 
using the known starting mass of each sample to calculate the approximate cell 
counts per gram of each microbe in the input sample material (before gDNA extraction).
These calculations are outlined in the figure below.

![pySynDNA OGU cell counts workflow](https://raw.githubusercontent.com/biocore/pysyndna/main/docs/absolute_quant_calc_cell_counts_workflow.png?raw=true)


## Installation

To install this package, first clone the repository from GitHub:

```
git clone https://github.com/biocore/pysyndna.git
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
`fit_linear_regression_models` function in the `pysyndna` module 
(or, for qiita-based usage, the `fit_linear_regression_models_for_qiita` 
function).

Second, apply these models to counts of microbial genomes in the 
samples by calling the `calc_ogu_cell_counts_biom` function in the 
`pysyndna` module, specifying whether to return the cell counts 
per gram of gDNA or the cell counts per gram of un-extracted sample material 
(the latter is usually the more relevant metric).  For qiita-based usage, 
call the `calc_ogu_cell_counts_per_g_of_sample_for_qiita` function (which 
always returns the cell counts per gram of un-extracted sample material.)

The documentation for these functions offer details on the specifics of 
their parameters, and the unit tests provide functional examples of inputs 
and outputs.