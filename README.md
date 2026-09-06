# Neural Operator PDE Surrogate Benchmark

A configurable PyTorch benchmark for **1D and 2D heat-equation surrogate modelling**.  
The project compares **MLP**, **CNN**, and **Fourier Neural Operator-style (FNO)** models for predicting PDE solution fields, with both **in-distribution** and **out-of-distribution (OOD)** evaluation.

The full workflow is:

```text
finite-difference PDE solver → simulation dataset → neural surrogate models → ID/OOD evaluation → model comparison
```

## Why this project?

Scientific and engineering simulations can be expensive to run repeatedly. Neural surrogate models can approximate simulator outputs much faster after training, but they must be evaluated carefully for accuracy and generalization.

This project demonstrates a compact simulation-to-ML workflow:

- finite-difference data generation for 1D and 2D heat equations
- configurable command-line training pipeline
- MLP, CNN, and FNO-style PyTorch models
- in-distribution and OOD testing on unseen diffusion coefficients
- automated metrics, plots, and comparison tables

## Problem formulation

The heat equation is

```math
\frac{\partial u}{\partial t}
=
\alpha \nabla^2 u
```

where `u` is the temperature field, `alpha` is the diffusion coefficient, and `∇²u` is the spatial Laplacian.

The surrogate modelling task is:

```math
u(0), \alpha \rightarrow u(T)
```

Given an initial temperature field and a diffusion coefficient, the model predicts the final temperature field after a fixed simulation time.

## Supported benchmarks

| Benchmark | Input field | Output field | Models |
|---|---:|---:|---|
| 1D heat equation | `(128,)` | `(128,)` | MLP, CNN1D, FNO1D |
| 2D heat equation | `(32, 32)` | `(32, 32)` | MLP, CNN2D, FNO2D |

## Dataset

The datasets are generated using explicit finite-difference solvers.

| Split | Samples | Diffusion coefficient range | Purpose |
|---|---:|---:|---|
| Train | 2000 | 0.005–0.030 | Model training |
| Validation | 300 | 0.005–0.030 | Checkpoint selection |
| Test | 300 | 0.005–0.030 | In-distribution evaluation |
| OOD test | 300 | 0.035–0.060 | Out-of-distribution evaluation |

The OOD test set uses larger diffusion coefficients than those seen during training.

## Results

### 1D heat equation

| Model | Test relative L2 error | OOD relative L2 error | Parameters |
|---|---:|---:|---:|
| MLP | 1.54% | 5.47% | 197,760 |
| CNN1D | 1.20% | 8.47% | 62,657 |
| FNO1D | **0.30%** | **3.01%** | 287,489 |

### 2D heat equation

| Model | Test relative L2 error | OOD relative L2 error | Parameters |
|---|---:|---:|---:|
| MLP | 6.83% | 6.90% | 1,575,936 |
| CNN2D | 0.50% | 3.67% | 312,257 |
| FNO2D | **0.45%** | **3.45%** | 1,188,385 |

## Key observations

- FNO-style models achieved the best overall performance in both 1D and 2D benchmarks.
- CNN models strongly outperformed MLPs on 2D fields, showing the importance of spatial inductive bias.
- In 1D, CNN improved in-distribution performance over MLP but generalized worse on OOD diffusion coefficients.
- The 2D CNN and FNO models both performed well, with FNO giving the best results.
- OOD evaluation is important because in-distribution accuracy alone does not fully describe surrogate reliability.

## Example outputs

After running training and comparison scripts, the project generates figures such as:

```markdown
![1D model comparison](plots/heat1d/model_comparison_relative_l2.png)

![2D model comparison](plots/heat2d/model_comparison_relative_l2.png)

![2D FNO predictions](plots/heat2d/fno_test_predictions.png)

![2D FNO OOD predictions](plots/heat2d/fno_ood_predictions.png)
```

## Installation

```bash
git clone https://github.com/adiManethia/neural-operator-pde-benchmark.git
cd neural-operator-pde-benchmark
pip install -e .
```

For GPU support, install the PyTorch build compatible with your CUDA version.

## Usage

Generate datasets:

```bash
python -m neural_pde.generate_data --dim 1
python -m neural_pde.generate_data --dim 2
```

Train models:

```bash
python -m neural_pde.train --dim 1 --model mlp
python -m neural_pde.train --dim 1 --model cnn
python -m neural_pde.train --dim 1 --model fno

python -m neural_pde.train --dim 2 --model mlp
python -m neural_pde.train --dim 2 --model cnn
python -m neural_pde.train --dim 2 --model fno
```

Compare models:

```bash
python -m neural_pde.compare --dim 1
python -m neural_pde.compare --dim 2
```

## Repository structure

```text
configs/                 YAML experiment configs
src/neural_pde/          main Python package
src/neural_pde/solvers/  finite-difference heat-equation solvers
src/neural_pde/models/   MLP, CNN, and FNO-style model definitions
data/                    generated datasets, ignored by git
checkpoints/             trained model checkpoints, ignored by git
results/                 metric tables and comparison CSV files
plots/                   generated prediction and comparison figures
reports/                 technical LaTeX report
notes/                   project notes and interview preparation
```

## Technical details

The FNO-style model uses spectral convolution. The field is transformed to Fourier space, selected low-frequency modes are multiplied by learned complex weights, and the result is transformed back to physical space.

For a spectral layer, the core operation is:

```math
Y_{b,o,m} = \sum_i X_{b,i,m} W_{i,o,m}
```

where `b` is the batch index, `i` is the input channel, `o` is the output channel, and `m` is the Fourier mode.

## Skills demonstrated

- Scientific machine learning
- PDE surrogate modelling
- PyTorch training pipelines
- Neural operator concepts
- Finite-difference simulation
- 1D and 2D field prediction
- OOD generalization testing
- Model comparison and scientific visualization
- Configurable Python package design

## Limitations and future work

Current limitations:

- the benchmark uses the heat equation, which is linear and smooth
- the 2D grid is relatively small at `32 × 32`
- only diffusion-coefficient OOD shift is tested
- no uncertainty estimation is included yet
- no active learning loop is included yet

Possible future extensions:

- 2D grid scaling to `64 × 64`
- Burgers equation or reaction-diffusion equation
- multi-time-step prediction
- uncertainty estimation with ensembles
- active learning for simulation selection
- inference runtime comparison against the finite-difference solver

## Technical report

A detailed LaTeX report can be included in `reports/`, covering the finite-difference solver, surrogate formulation, model architectures, spectral convolution, ID/OOD evaluation, and result analysis.
