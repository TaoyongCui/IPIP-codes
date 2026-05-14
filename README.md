# IPIP: Iterative Pretraining Framework for Interatomic Potentials

Official implementation for the paper:
**"Iterative Pretraining Framework for Interatomic Potentials"**
[ArXiv:2507.20118 (2025)](https://www.arxiv.org/abs/2507.20118)

If you find this code useful, please cite our work.

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Automated Pipeline](#automated-pipeline)
  - [Pipeline Stages](#pipeline-stages)
  - [Run the Full Pipeline](#run-the-full-pipeline)
  - [Pipeline Parameters](#pipeline-parameters)
  - [Test Mode](#test-mode)
- [Manual Usage](#manual-usage)
  - [1. Generate Pretraining Data](#1-generate-pretraining-data)
  - [2. Training](#2-training)
  - [3. Model Evaluation](#3-model-evaluation)
  - [4. Molecular Dynamics Simulation](#4-molecular-dynamics-simulation)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)

---

## Overview

Machine Learning Interatomic Potentials (MLIPs) enable *ab initio*-level accuracy for Molecular Dynamics (MD) simulations at a fraction of the computational cost. However, their success often depends on large labeled datasets and extensive training.

**IPIP (Iterative Pretraining for Interatomic Potentials)** addresses these challenges by introducing:

- **Iterative self-improvement** via cyclic pretraining and fine-tuning.
- **Forgetting mechanisms** to avoid convergence to suboptimal minima.
- **Cross-relabeling strategy** where the student model relabels in-domain data and the teacher model relabels OOD data.
- **Lightweight architectures** (PaiNN) that retain system-specific accuracy while improving efficiency.

Unlike general-purpose foundation models that trade accuracy for generality, IPIP achieves **>80% reduction in prediction error** and **up to 4x speedup** in the challenging **Mo-S-O** chemical system, enabling fast and accurate MD simulations.

---

## System Requirements

### Hardware

- NVIDIA GPU (tested on RTX A800); CPU-only mode available for testing
- Minimum 12 GB VRAM recommended for production runs

### Software

- **Operating System:** Linux (tested on Ubuntu 20.04)
- **Python:** 3.10.14
- **PyTorch:** 2.2.1 (CUDA 12.1)

---

## Installation

Create and activate the conda environment:

```bash
conda create -y -n ipip python=3.10.14
conda activate ipip
```

Install dependencies:

```bash
pip install torch==2.2.1 pytorch-lightning==2.4.0 torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install -U 'wandb>=0.12.10'
pip install ase mace-torch
```

---

## Quick Start

Verify your environment and run a complete end-to-end test with synthetic data (no GPU required):

```bash
python test_pipeline.py
```

This generates 10 synthetic data points and runs 2 iterations of the full IPIP pipeline on CPU in ~30 seconds.

---

## Automated Pipeline

The automated pipeline (`run_ipip_pipeline.py`) orchestrates the entire IPIP workflow, handling data flow, checkpoint management, and convergence detection across multiple iterations.

### Pipeline Stages

Each iteration executes the following 6 stages:

```
Stage 1 (once): Generate initial pretraining data via MD with teacher model (MACE-OFF)
     |
     v
Stage 2: Pretrain student model on pseudo-labeled data (force-only loss)
     |
     v
Stage 3: Finetune student model on DFT data (energy + force loss),
         initialized from the pretrained checkpoint
     |
     v
Stage 4: Run MD simulations with finetuned model to collect OOD configurations
     |
     v
Stage 5: Relabel and update pretraining data (cross-relabeling strategy)
     |
     v
Stage 6: Evaluate model and check convergence
     |
     +---> If converged: stop. Otherwise: repeat from Stage 2.
```

### Run the Full Pipeline

**Default (3 iterations):**

```bash
python run_ipip_pipeline.py --iterations 3 --finetune-data ./datasets/finetune_data.pt
```

**Custom configuration:**

```bash
python run_ipip_pipeline.py \
    --iterations 5 \
    --results-dir ./my_results \
    --finetune-data ./datasets/my_dft_data.pt \
    --pretrain-seeds 10 \
    --md-seeds-per-iter 5 \
    --convergence-threshold 0.01 \
    --data-retention-rate 0.5
```

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 3 | Number of refinement iterations |
| `--results-dir` | `./ipip_results` | Output directory for all results |
| `--project-dir` | `.` | Project root containing scripts |
| `--finetune-data` | `./datasets/finetune_data.pt` | Path to DFT-labeled finetuning data |
| `--pretrain-seeds` | 10 | Number of MD seeds for initial data generation |
| `--md-seeds-per-iter` | 5 | Number of MD seeds per iteration |
| `--convergence-threshold` | 0.01 | Stop when relative Force MAE improvement < threshold |
| `--data-retention-rate` | 0.5 | Fraction of old pretraining data to retain each iteration |
| `--test-mode` | off | Run with synthetic data on CPU |
| `--test-num-samples` | 10 | Number of synthetic samples in test mode |

### Test Mode

Test mode bypasses external dependencies (MACE-OFF, GPU) and validates the pipeline logic end-to-end using synthetic data:

```bash
python run_ipip_pipeline.py --iterations 2 --test-mode
```

In test mode the pipeline:
- Generates synthetic `torch_geometric.data.Data` objects instead of running MD
- Trains on CPU with minimal epochs (`--max-epochs 2`, `--limit-train-batches 2`)
- Disables WandB logging (uses CSV logger)
- Evaluates the model directly without `inference.py`

---

## Manual Usage

You can also run each stage independently for more control.

### 1. Generate Pretraining Data

Generate the pretraining MD data for the **Mo-S-O** system using the MACE-OFF teacher model:

```bash
bash Generate_Pretrain_data.sh
```

Then process trajectories into dataset format:

```bash
jupyter notebook traj2data.ipynb
```

### 2. Training

`train.py` supports both pretraining and finetuning via the `--pretrain` flag.

**Pretrain** (force-only loss):

```bash
python train.py --datadir ./datasets/pretrain.pt --pretrain True
```

**Finetune** (energy + force loss, with pretrained initialization):

```bash
python train.py \
    --datadir ./datasets/finetune.pt \
    --pretrain False \
    --load-ckpt ./checkpoint/pretrained/last.ckpt
```

**Full `train.py` options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--datadir` | `./datasets/transition-1x/` | Path to training data (`.pt` file) |
| `--pretrain` | `False` | `True` = force-only loss; `False` = energy + force loss |
| `--load-ckpt` | None | Load pretrained weights from a checkpoint |
| `--save-dir` | None | Custom checkpoint save directory |
| `--max-epochs` | 10000 | Maximum training epochs |
| `--accelerator` | `gpu` | `gpu` or `cpu` |
| `--strategy` | `ddp_find_unused_parameters_true` | PyTorch Lightning training strategy |
| `--batch-size` | 32 | Training batch size |
| `--num-workers` | 48 | Data loader workers |
| `--limit-train-batches` | 1600 | Max training batches per epoch |
| `--limit-val-batches` | 80 | Max validation batches per epoch |
| `--no-wandb` | off | Disable WandB, use CSV logger |

### 3. Model Evaluation

Compare baseline and self-trained models:

```bash
python inference.py \
    --baseline-path Baseline.ckpt \
    --model-path Selftraining2.ckpt \
    --test-data test.pt
```

This produces scatter plots, histograms, and box plots comparing Force MAE distributions.

Interactive evaluation notebooks are also available:

```bash
jupyter notebook test_finetune.ipynb       # Evaluate finetuned model
jupyter notebook test_in_realworld.ipynb   # Real-world transfer evaluation
```

### 4. Molecular Dynamics Simulation

Run MD simulations using a finetuned student model:

```bash
python Supp_traj_md.py --seed 0
```

---

## Project Structure

```
IPIP-codes-main/
├── run_ipip_pipeline.py       # Automated IPIP pipeline (main entry point)
├── train.py                   # Training script (pretrain / finetune)
├── training_module.py         # PyTorch Lightning training module
├── PaiNN.py                   # PaiNN model architecture
├── PAINN_Calculator.py        # ASE calculator wrapper for PaiNN
├── inference.py               # Model evaluation and comparison
├── Generate_Pretrain_data.py  # MD simulations with teacher model (MACE-OFF)
├── Supp_traj_md.py            # MD simulations with student model
├── utils.py                   # Utility functions
├── test_pipeline.py           # End-to-end pipeline test
├── traj2data.ipynb            # Trajectory to dataset conversion
├── test_finetune.ipynb        # Finetuning evaluation notebook
├── test_in_realworld.ipynb    # Real-world transfer evaluation
├── Selftraining_Supp.cif      # Initial structure for MD
├── run_ipip.sh                # Shell wrapper for pipeline
├── Generate_Pretrain_data.sh  # Shell wrapper for data generation
├── inference.sh               # Shell wrapper for inference
├── ipip_config_template.json  # Configuration template
├── requirements.txt           # Python dependencies
└── QUICK_REFERENCE.txt        # Quick reference card
```

## Outputs

When running the pipeline, results are saved to `--results-dir` with this structure:

```
ipip_results/
├── ipip_config.json                        # Pipeline configuration
├── pretrain_data/
│   ├── pretrain_iter_00.pt                 # Initial pretraining data
│   ├── pretrain_iter_01.pt                 # Updated data after iteration 1
│   └── ...
├── models/
│   ├── pretrain_iter_00/last.ckpt          # Pretrained model (iter 1)
│   ├── finetune_iter_00/last.ckpt          # Finetuned model (iter 1)
│   └── ...
├── md_trajectories/
│   ├── iteration_00/md_collected.pt        # MD trajectory data (iter 1)
│   └── ...
├── metrics/
│   ├── iteration_00_metrics.json           # Eval metrics (iter 1)
│   └── ...
└── logs/
```

Each `iteration_*_metrics.json` contains:

```json
{
  "iteration": 0,
  "energy_mae": 0.0762,
  "force_mae": 0.0451,
  "status": "completed",
  "pretrain_ckpt": "...",
  "finetune_ckpt": "...",
  "retained_count": 5,
  "new_data_count": 3,
  "combined_count": 8
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `--batch-size` or use `--accelerator cpu` |
| `FileNotFoundError` on finetune data | Ensure the `.pt` file path passed to `--finetune-data` exists |
| MD simulations diverge | Reduce timestep or temperature in `Supp_traj_md.py` |
| Force MAE not improving | Try more iterations or higher quality finetune data |
| WandB issues | Add `--no-wandb` to use CSV logger instead |
| Test on CPU | Use `--test-mode` or manually pass `--accelerator cpu` to `train.py` |

Monitor a running pipeline:

```bash
tail -f ipip_pipeline.log          # Live pipeline logs
watch nvidia-smi                    # GPU usage
cat ipip_results/metrics/*.json     # Iteration metrics
```

---

## License

This project is licensed under the **Apache License 2.0**.
For details, please refer to the [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

---

## Citation

If you use this repository, please cite:

```bibtex
@article{cui2025iterative,
  title={Iterative Pretraining Framework for Interatomic Potentials},
  author={Cui, Taoyong and Wang, Zhongyao and Zhou, Dongzhan and Li, Yuqiang and Bai, Lei and Ouyang, Wanli and Su, Mao and Zhang, Shufei},
  journal={arXiv preprint arXiv:2507.20118},
  year={2025}
}
```
