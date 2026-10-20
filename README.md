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

## Results (latest run)

On the public UCI splice-junction stand-in dataset (not epitope data), a class-balanced LightGBM reached 0.960 macro-F1 on the held-out test set versus 0.953 for a standard random forest on one-hot window features, a small gain from a single split.

| Metric | Value |
|---|---|
| baseline_score | 0.9531 |
| improved_score | 0.9596 |
| improvement | 0.006575 |
| baseline_score_std | 0.001182 |
| improved_score_std | 0 |
| baseline_accuracy | 0.9607 |
| improved_accuracy | 0.9662 |
| baseline_brier | 0.1249 |
| improved_brier | 0.06689 |
| brier_improvement | 0.05804 |
| cv_best_macro_f1 | 0.9569 |
| best_learning_rate | 0.1 |
| best_num_leaves | 15 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in task: DNA splice-junction windows, not linear B-cell epitopes; results say nothing about epitope prediction. Single split, so variance across splits is not measured. The improved model's final fit uses 80% of training rows (the rest calibrates isotonic). The baseline uses all training rows. Seeds vary model randomness only. The ucimlrepo internal HTTP calls cannot take a custom timeout; only the pre-flight request does.

## Reference

This project is motivated by the following paper:

> J Ashford, A Ekárt, F Campelo (2023). *Estimating the limits of organism-specific training for epitope prediction*. 2023 IEEE International. https://ieeexplore.ieee.org/abstract/document/10385381/

Paper author: Prof. Felipe Campelo, Aston University (f.campelo@aston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
