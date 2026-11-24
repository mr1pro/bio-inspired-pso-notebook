# Biologically Inspired Computation — MLP + PSO Coursework

A generic Multilayer Perceptron (MLP), implemented from scratch with NumPy,
trained by Particle Swarm Optimisation (PSO). Design follows `CW-design.pdf`:
Builder (MLP construction), Strategy (informant selection), Adapter
(connects PSO <-> MLP while keeping both independent).

## Start here

Open **`notebooks/MLP_PSO_Coursework.ipynb`** — it has markdown explanations
next to each code section, runs the full pipeline end to end, and already
contains the executed outputs/plots (open it on GitHub to see them without
running anything).

## Project layout

```
src/
  mlp/
    activations.py    # ActivationFunction, Sigmoid, Tanh, Softmax
    losses.py           # Loss, MSE, BinaryCrossEntropy, CategoricalCrossEntropy
    layer.py              # Layer: weights, biases, activation
    network.py             # Network: ordered layers, forward/predict, param vector
    builder.py               # MLPBuilder (Builder pattern)
  pso/
    fitness_function.py    # FitnessFunction interface
    particle.py              # Particle: position/velocity/personal best/informants
    informant_strategy.py    # Strategy pattern: Random vs NearestNeighbour
    population.py              # Population: main PSO loop
  adapter/
    mlp_fitness_adapter.py  # Adapter pattern: connects FitnessFunction <-> MLP
notebooks/
  MLP_PSO_Coursework.ipynb    # Main notebook: build, train, evaluate, compare
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/MLP_PSO_Coursework.ipynb
```

## What's already proven to work

- MLP forward pass, parameter flatten/restore, loss computation — NumPy only,
  no PyTorch/TensorFlow/Keras, no gradient descent in the network itself.
- PSO: particles, personal/global best, both informant-selection strategies,
  informants reassigned every iteration.
- Adapter: PSO never imports anything MLP-specific; training data only.
- Full pipeline run on `sklearn.datasets.load_iris()` as a placeholder
  dataset: test accuracy ≈ 0.97, convergence plots, and a Random vs
  Nearest-Neighbour informant-strategy comparison, all already executed
  in the notebook.

## Still to do

- [ ] Swap in the team's actual coursework dataset (`load_dataset()` in
      notebook section 1) — nothing else needs to change.
- [ ] Decide final hyperparameters and re-run the comparison cells with them.
- [ ] Write up the report (best training loss, test accuracy/precision/
      recall/F1, PSO convergence behaviour, hyperparameter + informant
      strategy comparison — see notebook section 9).
