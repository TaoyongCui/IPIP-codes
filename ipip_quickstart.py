#!/usr/bin/env python3
"""
Quick reference guide and validation script for IPIP pipeline
"""

import sys
import torch
import importlib
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed"""

    print("="*70)
    print("IPIP PIPELINE - DEPENDENCY CHECK")
    print("="*70)
    print()

    required_packages = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'pytorch_lightning': 'PyTorch Lightning',
        'ase': 'Atomic Simulation Environment',
        'numpy': 'NumPy',
        'mace': 'MACE',
    }

    all_installed = True

    for package, name in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name:30s} ({package:20s}) - v{version}")
        except ImportError:
            print(f"✗ {name:30s} ({package:20s}) - NOT INSTALLED")
            all_installed = False

    print()

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU Support:                  CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"⚠ GPU Support:                  CUDA not available (will use CPU)")

    print()

    # Check required files
    print("Required files:")
    required_files = {
        'train.py': 'Training module',
        'Generate_Pretrain_data.py': 'Data generation',
        'Supp_traj_md.py': 'MD simulation',
        'inference.py': 'Model inference',
        'training_module.py': 'Lightning module',
        'PaiNN.py': 'PaiNN model',
        'PAINN_Calculator.py': 'ASE calculator',
        'utils.py': 'Utility functions',
    }

    for filename, description in required_files.items():
        filepath = Path(filename)
        if filepath.exists():
            print(f"✓ {filename:30s} ({description})")
        else:
            print(f"✗ {filename:30s} ({description}) - NOT FOUND")
            all_installed = False

    print()

    if all_installed:
        print("✓ All dependencies and required files are available!")
        print()
        return True
    else:
        print("✗ Some dependencies or files are missing.")
        print()
        print("Install missing dependencies with:")
        print("  pip install -r requirements.txt")
        print()
        return False


def print_quick_start():
    """Print quick start guide"""

    print("="*70)
    print("IPIP PIPELINE - QUICK START GUIDE")
    print("="*70)
    print()

    guide = """
1. PREPARE YOUR DATA
   - Organize your DFT data as torch_geometric dataset
   - Save as: ./datasets/finetune_data.pt
   - Each sample should have:
     * pos: atomic coordinates (N × 3)
     * z: atomic numbers (N,)
     * energy: total energy (1,)
     * force: atomic forces (N × 3)

2. RUN THE PIPELINE (3 iterations)
   Option A: Python script
     python run_ipip_pipeline.py --iterations 3

   Option B: Bash script
     bash run_ipip.sh -i 3

3. MONITOR PROGRESS
   - Check logs: cat ipip_pipeline.log
   - Monitor GPU: nvidia-smi -l 1
   - View tensorboard: tensorboard --logdir checkpoint/ipip/

4. ANALYZE RESULTS
   - Check metrics: cat ipip_results/metrics/iteration_*_metrics.json
   - Trained models: ipip_results/models/
   - MD trajectories: ipip_results/md_trajectories/

5. ADJUST PARAMETERS (if needed)
   - Edit run_ipip_pipeline.py or run_ipip.sh
   - Key parameters:
     * --iterations: Number of refinement cycles
     * --md-seeds-per-iter: How many MD simulations per iteration
     * --data-retention-rate: How much old data to keep
     * --convergence-threshold: When to stop (early termination)
"""
    print(guide)


def print_command_reference():
    """Print command reference"""

    print("="*70)
    print("COMMAND REFERENCE")
    print("="*70)
    print()

    commands = """
PYTHON SCRIPT USAGE:
  python run_ipip_pipeline.py [OPTIONS]

OPTIONS:
  --iterations N                 Number of IPIP iterations (default: 3)
  --results-dir PATH             Output directory (default: ./ipip_results)
  --finetune-data FILE           Path to DFT dataset
  --pretrain-seeds N             Initial MD seeds (default: 10)
  --md-seeds-per-iter N          MD seeds per iteration (default: 5)
  --convergence-threshold FLOAT  Stop threshold (default: 0.01)
  --data-retention-rate FLOAT    Retention rate (default: 0.5)

BASH SCRIPT USAGE:
  bash run_ipip.sh [OPTIONS]

OPTIONS:
  -i, --iterations N
  -r, --results-dir PATH
  -f, --finetune-data FILE
  --md-seeds N
  --md-seeds-per-iter N
  -h, --help

EXAMPLES:
  # Basic run (3 iterations)
  python run_ipip_pipeline.py --iterations 3

  # Custom output directory
  python run_ipip_pipeline.py -i 5 -r ./my_results

  # Use custom data
  python run_ipip_pipeline.py -i 3 -f ./data/my_dft.pt

  # More MD samples per iteration
  bash run_ipip.sh -i 3 --md-seeds-per-iter 10

  # All options
  python run_ipip_pipeline.py \\
    --iterations 5 \\
    --results-dir ./ipip_run_v2 \\
    --finetune-data ./data/chignolin.pt \\
    --pretrain-seeds 20 \\
    --md-seeds-per-iter 10 \\
    --convergence-threshold 0.005 \\
    --data-retention-rate 0.6
"""
    print(commands)


def print_output_structure():
    """Print expected output directory structure"""

    print("="*70)
    print("OUTPUT DIRECTORY STRUCTURE")
    print("="*70)
    print()

    structure = """
ipip_results/
│
├── ipip_config.json                     # Pipeline configuration (reproducibility)
├── ipip_pipeline.log                    # Detailed execution log
│
├── pretrain_data/                       # Pretraining datasets
│   ├── iteration_00/                    # Initial data (teacher-generated)
│   ├── iteration_01/                    # Data for iteration 1
│   ├── iteration_02/                    # Data for iteration 2
│   └── ...
│
├── models/                              # Trained student models
│   ├── iteration_00/
│   │   ├── ff-epoch-XXX-*.ckpt         # Best checkpoint
│   │   └── ...
│   ├── iteration_01/
│   ├── iteration_02/
│   └── ...
│
├── md_trajectories/                     # MD simulation results
│   ├── iteration_00/
│   │   ├── traj_0.xyz                  # Trajectory from seed 0
│   │   ├── traj_1.xyz
│   │   └── ...
│   ├── iteration_01/
│   ├── iteration_02/
│   └── ...
│
├── metrics/                             # Performance metrics per iteration
│   ├── iteration_00_metrics.json        # Energy MAE, Force MAE, status
│   ├── iteration_01_metrics.json
│   ├── iteration_02_metrics.json
│   └── ...
│
└── logs/                                # PyTorch Lightning logs
    └── version_0/
        ├── events.out.tfevents.*
        └── hparams.yaml
"""
    print(structure)


def print_key_metrics():
    """Print key metrics to monitor"""

    print("="*70)
    print("KEY METRICS TO MONITOR")
    print("="*70)
    print()

    metrics = """
FOR EACH ITERATION, CHECK:

1. Force MAE (mean absolute error in force predictions)
   - Should decrease with each iteration (20-80% improvement expected)
   - If stagnating: convergence threshold may stop early

2. Energy MAE (mean absolute error in energy predictions)
   - Should also improve
   - May have larger initial variance

3. Simulation Stability
   - Proportion of stable MD trajectories
   - Should approach 100% after few iterations

4. Convergence Status
   - If improvement < threshold → pipeline stops
   - Default threshold: 1%

EXAMPLE PROGRESSION:
  Iteration 0: Force MAE = 0.50 eV/Å (baseline)
  Iteration 1: Force MAE = 0.42 eV/Å (↓ 16%)  → continue
  Iteration 2: Force MAE = 0.38 eV/Å (↓ 10%)  → continue
  Iteration 3: Force MAE = 0.37 eV/Å (↓ 3%)   → continue
  Iteration 4: Force MAE = 0.368 eV/Å (↓ 0.5%) → STOP (converged)

WHERE TO FIND METRICS:
  - During execution: tail -f ipip_pipeline.log
  - JSON results: cat ipip_results/metrics/iteration_*.json
  - Real-time: tensorboard --logdir checkpoint/ipip/
"""
    print(metrics)


def print_troubleshooting():
    """Print troubleshooting tips"""

    print("="*70)
    print("TROUBLESHOOTING")
    print("="*70)
    print()

    tips = """
COMMON ISSUES:

1. "CUDA out of memory"
   - Reduce batch size in train.py: bz = 16
   - Reduce num_workers: num_workers = 4
   - Reduce model size: hidden_channels = 64

2. "FileNotFoundError: No such file or directory"
   - Check data paths are correct
   - Verify ./datasets/finetune_data.pt exists
   - Verify current directory contains all .py files

3. "MD simulations diverge (NaN values)"
   - Reduce timestep: timestep = 0.5 fs
   - Reduce temperature: temp = 1500 K
   - Check initial structure is valid

4. "Force MAE not improving"
   - Increase num_iterations or --md-seeds-per-iter
   - Check finetune data quality
   - Verify initial models are properly trained

5. "Pipeline takes too long"
   - Reduce num_md_seeds_per_iter (default: 5)
   - Reduce num_iterations (default: 3)
   - Use fewer initial seeds (default: 10)
   - Consider early stopping (lower convergence_threshold)

FOR MORE HELP:
  - Check log: tail -100 ipip_pipeline.log
  - Review config: cat ipip_results/ipip_config.json
  - Inspect metrics: cat ipip_results/metrics/*.json
"""
    print(tips)


def main():
    """Main function"""

    print()

    # Check dependencies
    deps_ok = check_dependencies()

    print()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == '--quick-start':
            print_quick_start()
        elif command == '--commands':
            print_command_reference()
        elif command == '--output':
            print_output_structure()
        elif command == '--metrics':
            print_key_metrics()
        elif command == '--troubleshooting':
            print_troubleshooting()
        elif command == '--help':
            print_quick_start()
            print()
            print_command_reference()
            print()
            print_output_structure()
            print()
            print_key_metrics()
        else:
            print(f"Unknown command: {command}")
            print()
            print("Available commands:")
            print("  python ipip_quickstart.py --quick-start")
            print("  python ipip_quickstart.py --commands")
            print("  python ipip_quickstart.py --output")
            print("  python ipip_quickstart.py --metrics")
            print("  python ipip_quickstart.py --troubleshooting")
            print("  python ipip_quickstart.py --help (show all)")
    else:
        # Show everything
        print_quick_start()
        print()
        print_command_reference()
        print()
        print_output_structure()
        print()
        print_key_metrics()
        print()
        print_troubleshooting()

    print()

    if deps_ok:
        print("✓ You're ready to run IPIP!")
        print()
        print("Start with:")
        print("  python run_ipip_pipeline.py --iterations 3")
    else:
        print("✗ Please install missing dependencies first")
        print("  pip install -r requirements.txt")

    print()


if __name__ == '__main__':
    main()
