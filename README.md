# OAS HepB classification

This repository contains an analysis of `Galson 2015` and `Galson 2016` vaccination studies from the `Observed Antibody Space` database (http://opig.stats.ox.ac.uk/webapps/oas/) and pipeline focusing on preprocessing and prediction of HepB specificity of antibodies.

## Data

The full data can be downloaded from the official webpage of `Observed Antibody Space` - http://opig.stats.ox.ac.uk/webapps/oas/downloads.

## Makefile

The `Makefile` (`GNU Make`) contains targets for data preprocessing, data analysis and models' training. The `Makefile` targets have been executed on a high-performance computing cluster. Therefore, in the targets, internal `HPC` library responsible for handling the jobs on the cluster is being used.

## Python scripts

The `bin` folder contains python scripts responsible for data preprocessing and predictions as well as modules for visualizations and evaluations.

## Fairseq plugins

The `bin/fairseq_plugins` folder contains custom `fairseq` extensions such as the custom `RoBERTa` architecture that is used in this work.

## Notebooks

The `notebooks` folder contains `jupyter` notebooks with data preprocessing, data analysis, as well as evaluations of the models and representation visualizations of the data.

## Models

Fully-trained baselines in a zip file and checkpoints of `RoBERTa` models are available in the `models` folder.

## Text

The `text` folder contains thesis in `PDF` format and the `text/source` folder contains its corresponding source files.

## Other

The `setup.py` serves for python module installation and `environment.yml` is a `yaml` file listing required `conda` dependencies.
