# COLDSTART
### Commissioning Sample Complexity for Robotic Anomaly Detection

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Research](https://img.shields.io/badge/Status-Research-blue)](#)
[![Dataset](https://img.shields.io/badge/Datasets-voraus--AD%20%7C%20AURSAD-green)](#datasets)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</p>

---

## Overview

**COLDSTART** is a research framework for studying **Commissioning Sample Complexity** in robotic anomaly detection.

When a new industrial robot is commissioned, only a limited number of healthy operating cycles are typically available before production begins. This project investigates:

> **How many healthy commissioning cycles are required before an anomaly detector can be trusted in production?**

Unlike traditional anomaly-detection benchmarks that evaluate only AUROC or F1-score, COLDSTART evaluates detectors under realistic deployment constraints where:

- commissioning data are scarce,
- calibration must be leakage-safe,
- false alarms must remain extremely low,
- confidence intervals determine deployment readiness.

---

## Research Motivation

Industrial robots rarely have thousands of healthy commissioning examples available.

Instead, engineers often need to decide whether an anomaly detector is reliable after observing only a handful of healthy cycles.

This repository provides a reproducible benchmarking framework for answering that question through rigorous evaluation protocols.

---

# Key Features

- Leakage-safe commissioning protocol
- Frozen detector evaluation
- Multi-seed commissioning experiments
- Conformal threshold calibration
- Episode-level diagnostics
- Threshold fingerprint analysis
- Publication-quality figures
- Reproducible evaluation pipeline

---

# Repository Structure

```
COLDSTART/
│
├── data/                     # Dataset loading utilities
├── experiments/              # Experiment scripts
├── outputs/                  # Figures and evaluation outputs
├── src/                      # Core implementation
│   ├── detectors/
│   ├── evaluation/
│   ├── calibration/
│   ├── utils/
│   └── ...
│
├── notebooks/                # Exploratory analysis
├── docs/                     # Documentation
├── tests/                    # Unit tests
├── requirements.txt
└── README.md
```

---

# Supported Datasets

## voraus-AD

Industrial robotic pick-and-place anomaly detection dataset.

Used as the primary benchmark for commissioning experiments.

Official repository:

https://github.com/vorausrobotik/voraus-ad-dataset

---

## AURSAD

Robot screw-driving anomaly detection dataset.

Used for external validation of commissioning protocols.

---

# Research Questions

This project investigates:

- How many commissioning cycles are required before deployment?
- Does transfer learning reduce commissioning effort?
- Can shrinkage-based adaptation improve reliability?
- Are confidence intervals more informative than average performance?
- Which anomaly detectors remain trustworthy under limited healthy data?

---

# Current Evaluation Protocol

Each detector is evaluated using:

- Leakage-safe episode splits
- Frozen detector training
- Commissioning sizes

```
N = {10, 25, 50, 100, 250, 500}
```

Metrics include

- Recall
- False Positive Rate
- Success Rate
- Bootstrap Confidence Intervals
- Estimated commissioning requirement (N*)

---

# Implemented Methods

Current baselines include

- TargetOnly
- SourceOnly
- Pooled
- Isolation Forest
- Conformal kNN
- Phase-Aligned Conformal kNN (PAKCT)
- RACE
- Additional statistical baselines

---

# Reproducing Experiments

Clone the repository

```bash
git clone https://github.com/priestly-ops/COLDSTART.git

cd COLDSTART
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download datasets

```bash
python data/download_voraus.py
```

Run commissioning experiments

```bash
python experiments/run_voraus.py
```

Generate figures

```bash
python experiments/plot_commissioning_curves.py
```

---

# Current Status

### Completed

- Leakage-safe evaluation protocol
- Frozen detector pipeline
- Threshold fingerprint analysis
- Episode replacement diagnostics
- voraus-AD benchmark
- Statistical baselines
- Commissioning evaluation framework

### In Progress

- AURSAD evaluation
- Cross-dataset validation
- Publication figures
- Manuscript preparation

---

# Reproducibility

The framework is designed to be fully reproducible.

- Fixed random seeds
- Deterministic evaluation
- Leakage-safe data splitting
- Separate commissioning, calibration, and testing sets
- Bootstrap confidence intervals

---

# Results

Example outputs include

- Commissioning curves
- Threshold evolution
- Success-rate plots
- Confidence intervals
- Episode diagnostics
- Detector comparison tables

Generated figures are stored in

```
outputs/
```

---

# Project Roadmap

- [x] Leakage-safe commissioning framework
- [x] Frozen detector evaluation
- [x] Statistical baseline implementation
- [x] Threshold diagnostics
- [x] voraus-AD experiments
- [ ] AURSAD external validation
- [ ] Camera-ready manuscript
- [ ] Public benchmark release

---

# Citation

If this repository contributes to your research, please cite it once the accompanying paper becomes available.

A `CITATION.cff` file will be added upon publication.

---

# Acknowledgements

This work builds upon publicly available robotic anomaly-detection datasets, including:

- voraus-AD
- AURSAD

We thank the dataset authors for making their work publicly available to the research community.

---

# Disclaimer

This repository is intended for **research purposes only**.

The anomaly detectors implemented here are **not certified for safety-critical industrial deployment** and should not be used as the sole basis for operational or maintenance decisions.

---

# Contact

**Priestly Barigala**

MS Robotics • Colorado School of Mines

GitHub:
https://github.com/priestly-ops

---

## License

This project will be released under the **MIT License**.


