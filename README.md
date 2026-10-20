# Balanced LightGBM vs random forest on a residue-window stand-in task

Motivation: Linear B-cell epitope predictors are usually trained on residue windows, and the paper "Estimating the limits of organism-specific training for epitope prediction" (2023) asks how far such models can go when training data are limited to one organism. This experiment does not reproduce that study. It tests a simpler question on a public stand-in task, to see whether a class-balanced gradient-boosted model on one-hot window features can outperform a standard random forest.

Question: Can a class-balanced LightGBM with one-hot window features beat a standard random forest on a public residue-window sequence classification task that stands in for linear B-cell epitope prediction?

Dataset: UCI Molecular Biology Splice-junction Gene Sequences (fixed-length DNA windows, three classes), loaded with the `ucimlrepo` package. This is NOT the paper's own data and not epitope data.

Baseline: a standard 300-tree random forest on one-hot encoded window positions, using raw class probabilities.

Improved model: a class-balanced LightGBM on the same one-hot features, with a small cross-validated grid over learning rate and number of leaves, plus isotonic calibration fitted on an inner split of the training data.

Metric: macro-F1 on a stratified 75/25 test split (higher is better). Accuracy and multiclass Brier score are reported as secondary metrics.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (and `figures/macro_f1.png`).

## Reference

This project is motivated by the following paper:

> J Ashford, A Ekárt, F Campelo (2023). *Estimating the limits of organism-specific training for epitope prediction*. 2023 IEEE International. https://ieeexplore.ieee.org/abstract/document/10385381/

Paper author: Prof. Felipe Campelo, Aston University (f.campelo@aston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
