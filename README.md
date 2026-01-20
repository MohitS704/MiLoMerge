# MiLoMerge

[![arXiv](https://img.shields.io/badge/arXiv-2601.10822-b31b1b.svg)](https://arxiv.org/abs/2601.10822)
[![Homepage](https://img.shields.io/badge/Homepage-spin.pha.jhu.edu-blue.svg)](https://spin.pha.jhu.edu/)

![Python](https://img.shields.io/badge/python->=%203.8-blue.svg)
[![PyPI version](https://badge.fury.io/py/MiLoMerge.svg)](https://badge.fury.io/py/MiLoMerge)
[![Code style: black](https://img.shields.io/badge/code%20style-black-orange)](https://black.readthedocs.io/en/stable/)

A package to merge bins together in such a way that the separability between distributions is minimally lossless.
The ROC and LOC curves defined in the paper above are also included as functions in this package.
It is fully Pythonic, with full interoperability with Numpy.

This package is derived from the findings of [Maximizing Returns: Optimizing Experimental Observables at the LHC](https://arxiv.org/abs/2601.10822), which should
be cited should the package be used.

## Installation

### Through pip

MiLoMerge is available in pip, and soon to conda, and can be installed as below:

```bash
pip install MiLoMerge
```

### Manual Installation

Download the .tar.gz installation available
at [https://spin.pha.jhu.edu/](https://spin.pha.jhu.edu/)
and run the following command in the MiLoMerge directory:

```bash
pip install .
```

## Getting Started

To use MiLoMerge, import the package within your file, and 
generate distributions to merge. The documentation,
alongside useful examples and tutorials,
is available at the [homepage](https://spin.pha.jhu.edu/MiLoMerge/).