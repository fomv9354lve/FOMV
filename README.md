# FOMV: Field Operator for Measured Viability

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Code accompanying the paper **"FOMV: Field Operator for Measured Viability – A Geometric Framework for First-Passage Analysis in Nonlinear Stochastic Systems"** by Osvaldo Morales.

## 📄 About

This repository contains the simulation code for the HARD-nonlinear model presented in the paper. The code estimates the **committor** (probability of recovery before collapse) and the **mean first-passage time (MFPT)** to collapse for a 6-dimensional stochastic dynamical system representing institutional variables: backlog (B), memory (M), execution (E), governance (G), trust (T), and coherence (C).

## 🚀 Getting Started

### Prerequisites

#### Python (version 3.7+)
- numpy
- scipy
- matplotlib
- tqdm

Install with:
```bash
pip install numpy scipy matplotlib tqdm
