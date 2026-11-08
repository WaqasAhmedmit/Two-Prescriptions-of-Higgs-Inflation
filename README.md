# Two Prescriptions of Nonminimal Higgs Inflation: Deep Learning and Future CMB Tests

## Overview

This repository provides a numerical and machine-learning framework for investigating **radiatively corrected nonminimal Higgs inflation under two renormalization prescriptions**.

The project explores how nonminimal gravitational coupling and quantum radiative corrections affect inflationary predictions and their compatibility with current and future cosmic microwave background (CMB) observations.

The analysis focuses on the scalar spectral index \(n_s\), tensor-to-scalar ratio \(r\), running \(\alpha_s\), and the influence of radiative corrections.

## Repository Files

| File | Description |
|---|---|
| `Scan_Prescription_I.py` | Numerical parameter scan for Prescription I |
| `Scan_Prescription_II.py` | Independent numerical parameter scan for Prescription II |
| `Higgs_ML_Current_Future_Radiative.ipynb` | Numerical visualization, machine-learning analysis, and future CMB forecasts |

## Research Objectives

- Investigate the influence of positive and negative radiative corrections on both prescriptions.
- Compare inflationary predictions with ACT, SPT, Planck, and BICEP/Keck constraints.
- Reconstruct inflationary observables using multilayer perceptron (MLP) neural networks and other machine-learning methods.
- Explore whether future CMB experiments can distinguish between the prescriptions.
- Generate publication-quality plots of inflationary parameter spaces and predictions.

## Numerical Scanning

The numerical framework supports large-scale parameter scans over

\[
10^{-6}\leq\xi\leq10^5,\qquad
50\leq N_*\leq60.
\]

The radiative ratio \(\kappa/\lambda\) is sampled with positive and negative signs up to an absolute value of 0.05.

The code evaluates slow-roll observables, resolves inflationary trajectories, and saves numerical results for further analysis.

## Installation

Install the required Python packages:

```bash
pip install numpy scipy pandas matplotlib scikit-learn jupyter
```

## Running the Scans

Prescription I:

```bash
python Scan_Prescription_I.py --samples 300000 --workers 4
```

Prescription II:

```bash
python Scan_Prescription_II.py --samples 300000 --workers 4
```

For machine-learning analysis, place the generated `Prescription_I_ALL.dat` and `Prescription_II_ALL.dat` files inside a folder named `data` beside the Jupyter notebook.

## Machine Learning and Future CMB Analysis

The notebook includes Random Forest classification, ExtraTrees regression, and multilayer perceptron neural networks to investigate the relationship between inflationary parameters and observables.

It also studies the impact of radiative corrections and compares numerical tensor predictions with illustrative sensitivities of Simons Observatory, South Pole Observatory, and LiteBIRD.

The acceptance classifications and future sensitivity studies are conditional numerical comparisons, not direct measurements of statistical model preference.

## Outputs

The code generates numerical `.dat` files, publication-quality PDF and PNG figures, machine-learning validation plots, and tables of observational screening results.

## Author

**Waqas Ahmed**

Center for Fundamental Physics  
School of Artificial Intelligence  
Hubei Polytechnic University  
Huangshi, China

## Citation

If you use this software in academic research, please cite the associated publication once available and the relevant inflationary model references.
