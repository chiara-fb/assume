# Market Power

This repository contains the reference implementation for the paper:

> **Learning to exploit market power in electricity markets through portfolio-level bidding**  
> Fusar Bassini C., Miskiw K.K., Maurer F., Donti P.L., and Kaack L.H.
> Submitted to *EEM 2025*, Trondheim, Norway.  

---

## Overview

This code uses the RL-strategy for an electricity portfolio operator as described in the paper, and includes:

- A test scenario benchmarked against marginal cost bidding, MADRL, and bilevel optimization
- A larger scenario representing a portfolio operator in the German market

---

## Repository Structure

```
.
├── inputs/                     # Inputs for scenarios "base" and "germany"
├── germany_case.ipynb          # Run simulation for "germany"       
├── test_case.ipynb             # Run simulation for "base" 
├── util_funcs/
│   ├── bilevel_solution.py     # Compute optimal solution for "base"
│   ├── db_plot.py              # Plot simulation database
│   ├── db_read.py              # Read simulation database
│   ├── learning.py             # Seeding, hyperparameter tuning
│   ├── market_power_index.py   # Indices to measure market power
|   ├── paper_figures.py        # Generates paper figures
└── README.md

