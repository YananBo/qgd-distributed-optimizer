# QGD: Quantized Gradient Descent for Distributed Nonconvex Optimization

[![Paper](https://img.shields.io/badge/PNAS-2024-blue)](https://doi.org/10.1073/pnas.2319625121)
[![arXiv](https://img.shields.io/badge/arXiv-2403.10423-b31b1b)](https://arxiv.org/abs/2403.10423)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

A PyTorch implementation of **Quantized Gradient Descent (QGD)** — a communication-efficient distributed optimizer that exploits stochastic quantization to escape saddle points in nonconvex optimization. Published in *Proceedings of the National Academy of Sciences* (PNAS), 2024.

---

## Why QGD Matters
Distributed training is the backbone of modern ML infrastructure — but some problems keep showing up in production:

| Production Pain Point | How QGD Solves It |
|----------------------|-------------------|
| **Communication bottleneck** — gradient/parameter exchange dominates wall-clock time in multi-GPU and cross-datacenter training | QGD compresses all inter-node communication via stochastic quantization — adjustable from 32-bit down to just a few bits per parameter, with **provable convergence guarantees** |
| **Saddle points in nonconvex landscapes** — large models (LLMs, diffusion, recommender systems) have optimization landscapes riddled with saddle points that degrade model quality | Most approaches inject artificial noise (extra compute + tuning). QGD gets saddle-point escape **for free** from the quantization noise that's already there for compression |
| **Edge / federated deployment** — surveillance cameras, autonomous vehicles, IoT sensors can't centralize raw data due to bandwidth and privacy | Agents exchange only quantized model parameters; raw data never leaves the device |
| **No extra hyperparameters** — perturbed GD methods add noise scales, schedules, and warm-up stages to tune | QGD's perturbation comes from quantization itself — the only practical knob is quantization level, which directly maps to your bandwidth budget |


**Key insight**: Quantization noise — traditionally treated as a nuisance to suppress — is actually *exactly the right kind of perturbation* to escape saddle points. QGD is the first method to prove this and exploit it.

---

## Highlights

- **Saddle-point avoidance via quantization**: Switching quantization sets ensures persistent perturbation in every direction — provably escaping all strict saddle points
- **Communication efficiency**: Compression of inter-node communication with no accuracy loss
- **Convergence to second-order stationary points**: The first distributed quantized method with this guarantee
- **Drop-in PyTorch optimizer**: Implemented as a `torch.optim.Optimizer` subclass

---

## Experiments

| Experiment | Motivating Application | What This Code Demonstrates | Docs |
|-----------|------------------------|-----------------------------|------|
| [Neural Network](experiments/neural_network/) | Decentralized / distributed model training | QGD vs. 7 decentralized optimizer baselines on CIFAR-10/MNIST with 5 workers, IID and non-IID data splits | [README](experiments/neural_network/README.md) |
| [Robust PCA](experiments/robust_pca/) | Edge-video background subtraction | 5-agent distributed robust PCA simulation on Wallflower video frames; agents keep local frame shards and exchange quantized low-rank factors, with QGD and DGD implementations logging reconstruction error | [README](experiments/robust_pca/README.md) |
| [Tensor Decomposition](experiments/tensor_decomposition/) | Tensor factorization for scientific data, signal processing, and recommender-style workloads | QGD implementation for Tucker decomposition with 5-agent quantized consensus, even/odd stochastic quantization, stepsize holding, and CSV logging of per-agent losses and reconstruction metrics | [README](experiments/tensor_decomposition/README.md) |
| [Binary Classification](experiments/binary_classification/) | Toy saddle-point diagnostic | 2-D regularized logistic-loss landscape for visualizing quantization-induced perturbations and comparing QGD with vanilla DGD trajectories | [README](experiments/binary_classification/README.md) |
## Quick Start

```bash
# Clone
git clone https://github.com/YananBo/qgd-distributed-optimizer.git
cd qgd-distributed-optimizer

# Install
pip install -r requirements.txt

# Run a quick demo — watch QGD escape a saddle point (~30 seconds)
cd experiments/binary_classification
python binary_classification.py --method qgd --iterations 5000

# Compare against vanilla DGD (gets stuck at the saddle point)
python binary_classification.py --method dgd --iterations 5000
```

---

## Repository Structure

```
├── experiments/
│   ├── neural_network/          # Distributed CNN training (CIFAR-10 / MNIST)
│   ├── robust_pca/              # Video background separation across edge cameras
│   ├── tensor_decomposition/    # Distributed Tucker decomposition
│   └── binary_classification/   # 2-D saddle-point escape demo
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Each experiment has its **own README** with CLI reference, hyperparameter tables, and architecture diagrams.

---

## Requirements

```
torch>=1.10
torchvision>=0.11
numpy>=1.21
tensorly>=0.7
scikit-learn>=1.0
pandas>=1.3
Pillow>=8.0
```

---

## Citation

```bibtex
@article{bo2024quantization,
  title     = {Quantization Avoids Saddle Points in Distributed Optimization},
  author    = {Bo, Yanan and Wang, Yongqiang},
  journal   = {Proceedings of the National Academy of Sciences},
  volume    = {121},
  number    = {17},
  pages     = {e2319625121},
  year      = {2024},
  publisher = {National Academy of Sciences},
  doi       = {10.1073/pnas.2319625121}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
