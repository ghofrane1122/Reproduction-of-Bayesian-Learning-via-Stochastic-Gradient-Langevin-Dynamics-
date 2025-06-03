# Bayesian Learning via Stochastic Gradient Langevin Dynamics (SGLD)

This repository presents a comprehensive reproduction and analysis of the seminal work by Welling and Teh (2011) on **Stochastic Gradient Langevin Dynamics (SGLD)**. The aim is to validate the theoretical foundation of the algorithm and demonstrate its practical effectiveness in scalable Bayesian inference across a range of machine learning tasks.

---

## 📄 Overview

SGLD is a sampling algorithm that combines the computational efficiency of stochastic gradient descent (SGD) with the probabilistic rigor of Langevin dynamics. Unlike traditional MCMC methods, SGLD operates with mini-batches and avoids Metropolis-Hastings corrections, making it ideal for large datasets.

This implementation reproduces and analyzes three key experiments from the original paper:

- **Gaussian Mixture Model (GMM):** Testing multimodal posterior exploration.
- **Bayesian Logistic Regression:** High-dimensional Bayesian inference on real-world data.
- **Independent Component Analysis (ICA):** Unsupervised learning with uncertainty quantification.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `Preliminaries_SGLD.ipynb` | Introductory notebook with background and setup |
| `FinalVersionOfSection5_1.ipynb` | Gaussian Mixture Model experiment |
| `FinalVersionOfSection5_2.ipynb` | Bayesian Logistic Regression experiment |
| `FinalVersionOfSection5_3.ipynb` | ICA (Independent Component Analysis) experiment |
| `README.md` | Project documentation (this file) |

> The file `sgld_detailed_complete.py` has been removed in the latest commit.

---

## 🔬 Experiments Summary

### 1. Gaussian Mixture Model (GMM)
- Demonstrates SGLD’s ability to explore multimodal distributions.
- Successfully captures both posterior modes, validating theoretical expectations.

### 2. Bayesian Logistic Regression
- Conducted on the UCI Adult dataset (`a9a`).
- SGLD enables sparse inference and provides posterior uncertainty over model coefficients.

### 3. Independent Component Analysis (ICA)
- Applied in both 2D and 6D source separation tasks.
- Compared with corrected Langevin sampling.
- SGLD shows stronger posterior exploration and better component instability profiles.

---

## Authors
This project was completed by:

Ghofrane Barouni

Sanda Dhouib
