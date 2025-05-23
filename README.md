This repository contains the code and experiments for the reproduction of the paper:

> **Bayesian Learning via Stochastic Gradient Langevin Dynamics**  
> *Max Welling & Yee Whye Teh, ICML 2011*

---

## Overview

The goal of this project is to implement and validate the Stochastic Gradient Langevin Dynamics (SGLD) algorithm proposed in the paper. The authors introduced SGLD as a method for performing approximate Bayesian inference efficiently with large datasets, using mini-batch stochastic gradients combined with injected noise.

The paper evaluated SGLD on three types of models:

1. **Mixture of Gaussians** – Demonstrates posterior sampling in a bimodal setting.
2. **Logistic Regression** – Applies SGLD to a real-world classification problem (a9a dataset).
3. **Independent Component Analysis (ICA)** – Uses SGLD with natural gradients to separate latent sources.

---
