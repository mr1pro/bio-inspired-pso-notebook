# Training a neural network with particle swarm optimisation (notebook)

F21BC Biologically Inspired Computation coursework. A feedforward neural
network and particle swarm optimisation (PSO), both written from scratch in
NumPy, are coupled so that PSO finds the network's weights and biases. The
network learns to tell genuine from forged banknotes
([UCI banknote authentication data](https://archive.ics.uci.edu/dataset/267/banknote+authentication)).

This repository is the notebook version: one walk-through notebook on top of
the same `src/` package as the companion repository
[`bio-inspired-coursework`](https://github.com/mr1pro/bio-inspired-coursework),
which holds the command-line scripts, the unit tests and the full experimental
study.

## Start here

Open **`notebooks/MLP_PSO_Coursework.ipynb`**. It is already executed, so the
outputs and plots can be read on GitHub without running anything. It covers:

1. the data and the stratified 70/30 split
2. building the network from a configuration
3. how a PSO particle becomes a network and gets its fitness
4. PSO following Algorithm 39 of *Essentials of Metaheuristics*
5. training, convergence and test metrics
6. small experiments (10 runs each): coefficients, architecture, activation,
   swarm size against iterations, informant selection
7. PSO choosing the activation function

## Running it

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
jupyter notebook notebooks/MLP_PSO_Coursework.ipynb
```

The whole notebook runs in under a minute.

## Layout

```
notebooks/MLP_PSO_Coursework.ipynb   the walk-through
src/mlp/        the network: activations, Layer, Network, MLPBuilder, losses
src/pso/        PSO: Particle, Population (Algorithm 39), informant strategies, boundary handling
src/adapter/    the coupling: particle -> network weights -> training accuracy
src/data/       loading, stratified split, k-fold, standardisation
src/evaluation/ accuracy, precision, recall, F1
data/           the UCI data file (Lohweg, 2012, CC BY 4.0)
report/         the coursework report (Word and PDF)
archive/iris_draft/   the earlier draft report on the Iris data, kept for reference only
```

`src/` is an exact copy of the package in `bio-inspired-coursework`.

The other documents in the top folder (`MLP_PSO_Study_Guide.pdf`,
`Report_Writing_Guide.docx`, `IEEE_Report_Template_BLANK.docx`) are earlier
planning material and are not part of the submission.
