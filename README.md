# IPIP: Iterative Pretraining Framework for Interatomic Potentials

Official implementation for the paper:  
**"Iterative Pretraining Framework for Interatomic Potentials"**  
📄 [ArXiv:2507.20118 (2025)](https://www.arxiv.org/abs/2507.20118)

If you find this code useful, please cite our work.

---

## 📘 Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Generate Pretraining Data](#1-generate-pretraining-data)
  - [2. Pretraining and Fine-tuning](#2-pretraining-and-fine-tuning)
  - [3. Model Evaluation](#3-model-evaluation)
  - [4. Molecular Dynamics Simulation](#4-molecular-dynamics-simulation)
- [License](#license)
- [Citation](#citation)

---

## 🧠 Overview

Machine Learning Interatomic Potentials (MLIPs) enable *ab initio*-level accuracy for Molecular Dynamics (MD) simulations at a fraction of the computational cost. However, their success often depends on large labeled datasets and extensive training.

**IPIP (Iterative Pretraining for Interatomic Potentials)** addresses these challenges by introducing:
- **Iterative self-improvement** via cyclic pretraining and fine-tuning.
- **Forgetting mechanisms** to avoid convergence to suboptimal minima.
- **Lightweight architectures** that retain system-specific accuracy while improving efficiency.

Unlike general-purpose foundation models that trade accuracy for generality, IPIP achieves **>80% reduction in prediction error** and **up to 4× speedup** in the challenging **Mo–S–O** chemical system, enabling fast and accurate MD simulations.

---

## ⚙️ System Requirements

### Hardware
- NVIDIA GPU (tested on RTX A800)
- Minimum 12 GB VRAM recommended

### Software
- **Operating System:** Linux (tested on Ubuntu 20.04)
- **Python:** 3.10.14
- **PyTorch:** 2.2.1 (CUDA 12.1)

---

## 🧩 Installation

Create and activate the conda environment:

```bash
conda create -y -n ipip python=3.10.14
conda activate ipip
```

Install dependencies:

```bash
pip install torch==2.2.1 pytorch-lightning==2.4.0 torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv   -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install -U 'wandb>=0.12.10'
```

---

## 🚀 Usage

### 1. Generate Pretraining Data

Generate the pretraining MD data for the **Mo–S–O** system:

```bash
sh Generate_Pretrain_data.sh
```

Then process trajectories into dataset format using the notebook:

```bash
traj2data.ipynb
```

---

### 2. Pretraining and Fine-tuning

Pretrain the model:

```bash
python train.py --datadir ./datasets/pretrain.pt --pretrain True
```

Fine-tune the pretrained model:

```bash
python train.py --datadir ./datasets/finetune.pt --pretrain False
```

---

### 3. Model Evaluation

Evaluate the fine-tuned model:

```bash
test_finetune.ipynb
```

Evaluate in real-world transfer settings:

```bash
test_in_realworld.ipynb
```

---

### 4. Molecular Dynamics Simulation

Run molecular dynamics simulations using the fine-tuned model:

```bash
python Supp_traj_md.py
```

---

## 📄 License

This project is licensed under the **Apache License 2.0**.  
For details, please refer to the [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

---

## 🧭 Citation

If you use this repository, please cite:

```bibtex
@article{cui2025iterative,
  title={Iterative Pretraining Framework for Interatomic Potentials},
  author={Cui, Taoyong and Wang, Zhongyao and Zhou, Dongzhan and Li, Yuqiang and Bai, Lei and Ouyang, Wanli and Su, Mao and Zhang, Shufei},
  journal={arXiv preprint arXiv:2507.20118},
  year={2025}
}
```
