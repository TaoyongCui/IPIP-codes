#!/usr/bin/env python3
"""
IPIP (Iterative Pretraining Framework for Interatomic Potentials) - Automated Pipeline

This script automates the complete IPIP workflow:
1. Initial pretraining data generation
2. Pretraining student model with teacher guidance
3. Finetuning on DFT data
4. MD simulation to collect OOD/edge cases
5. Iterative relabeling and refinement

Usage:
    python run_ipip_pipeline.py --iterations 3 --pretrain-data ./datasets/pretrain.pt --finetune-data ./datasets/finetune.pt
    python run_ipip_pipeline.py --iterations 2 --test-mode   # quick test with synthetic data
"""

import argparse
import os
import sys
import json
import logging
import subprocess
import re
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from typing import Dict, Any, Optional, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ipip_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(num_samples: int = 10, seed: int = 42) -> list:
    """Generate synthetic torch_geometric Data objects for testing."""
    from torch_geometric.data import Data
    rng = np.random.RandomState(seed)
    data_list = []
    for _ in range(num_samples):
        num_atoms = rng.randint(4, 8)
        cell_size = 10.0
        pos = torch.tensor(rng.rand(num_atoms, 3) * cell_size * 0.5 + cell_size * 0.25, dtype=torch.float32)
        z = torch.tensor(rng.choice([1, 6, 7, 8], size=num_atoms), dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float32) * cell_size
        energy = torch.tensor([rng.randn() * 0.1], dtype=torch.float32)
        force = torch.tensor(rng.randn(num_atoms, 3) * 0.01, dtype=torch.float32)
        natoms = torch.tensor([num_atoms], dtype=torch.long)
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        data = Data(
            pos=pos, z=z, cell=cell, energy=energy,
            force=force, natoms=natoms, pbc=pbc,
        )
        data_list.append(data)
    return data_list


class IPIPPipeline:
    """Manages the complete IPIP workflow."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.iteration = 0
        self.results_dir = Path(config.get('results_dir', './ipip_results')).resolve()
        self.project_dir = Path(config.get('project_dir', '.')).resolve()
        self.test_mode = config.get('test_mode', False)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_history: Dict[str, str] = {}

        self.setup_directories()
        self.save_config()

        logger.info(f"IPIP Pipeline initialized with {config['num_iterations']} iterations")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Project directory: {self.project_dir}")
        if self.test_mode:
            logger.info("*** RUNNING IN TEST MODE ***")

    def setup_directories(self):
        for dir_name in ['pretrain_data', 'models', 'md_trajectories', 'logs', 'metrics']:
            (self.results_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def save_config(self):
        config_path = self.results_dir / 'ipip_config.json'
        config_to_save = {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()}
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)

    def _run_command(self, cmd: List[str], description: str) -> subprocess.CompletedProcess:
        """Run a subprocess command with logging and error handling."""
        cmd_str = ' '.join(cmd)
        logger.info(f"Running: {cmd_str}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(self.project_dir)
        )
        if result.returncode != 0:
            logger.error(f"{description} failed (exit code {result.returncode})")
            logger.error(f"STDOUT:\n{result.stdout[-2000:] if result.stdout else '(empty)'}")
            logger.error(f"STDERR:\n{result.stderr[-2000:] if result.stderr else '(empty)'}")
            raise RuntimeError(f"{description} failed with exit code {result.returncode}")
        logger.info(f"{description} completed successfully")
        return result

    def _find_best_checkpoint(self, save_dir: str) -> Optional[str]:
        """Find the best checkpoint from a training run."""
        info_path = os.path.join(save_dir, 'training_info.json')
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            best = info.get('best_checkpoint', '')
            if best and os.path.exists(best):
                return best

        ckpt_files = list(Path(save_dir).rglob('*.ckpt'))
        if not ckpt_files:
            logger.warning(f"No checkpoint found in {save_dir}")
            return None
        best = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        return str(best)

    def _get_pretrain_data_path(self, iteration: int) -> str:
        """Get the pretrain data file path for a given iteration."""
        if iteration == 0:
            return str(self.results_dir / 'pretrain_data' / 'pretrain_iter_00.pt')
        return str(self.results_dir / 'pretrain_data' / f'pretrain_iter_{iteration:02d}.pt')

    def _get_train_args(self) -> List[str]:
        """Get extra train.py arguments for test mode."""
        if self.test_mode:
            return [
                '--max-epochs', '2',
                '--accelerator', 'cpu',
                '--strategy', 'auto',
                '--num-workers', '0',
                '--batch-size', '4',
                '--limit-train-batches', '2',
                '--limit-val-batches', '1',
                '--no-wandb',
            ]
        return []

    # ========================== STAGE 1 ==========================
    def generate_initial_pretrain_data(self) -> str:
        """Stage 1: Generate initial pretraining dataset using teacher model."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: Generating Initial Pretraining Data")
        logger.info("=" * 80)

        output_path = self._get_pretrain_data_path(0)

        if self.test_mode:
            logger.info("Test mode: generating synthetic pretrain data")
            data_list = generate_synthetic_data(
                num_samples=self.config.get('test_num_samples', 10), seed=42
            )
            torch.save(data_list, output_path)
            logger.info(f"Saved {len(data_list)} synthetic samples to {output_path}")
            return output_path

        logger.info(f"Running {self.config['num_md_seeds']} MD simulations with teacher model...")
        for seed in range(self.config['num_md_seeds']):
            logger.info(f"  MD simulation {seed + 1}/{self.config['num_md_seeds']} (seed={seed})")
            cmd = [sys.executable, str(self.project_dir / 'Generate_Pretrain_data.py'), '--seed', str(seed)]
            try:
                self._run_command(cmd, f"MD simulation seed={seed}")
            except RuntimeError:
                logger.warning(f"  MD simulation {seed} failed, continuing...")

        if self.config.get('pretrain_data_source'):
            import shutil
            shutil.copy(self.config['pretrain_data_source'], output_path)
            logger.info(f"Copied pretrain data from {self.config['pretrain_data_source']}")
        else:
            logger.warning("No pretrain_data_source configured; collect MD outputs manually "
                           f"and save to {output_path}")

        return output_path

    # ========================== STAGE 2 ==========================
    def pretrain_student_model(self, iteration: int) -> Optional[str]:
        """Stage 2: Pretrain lightweight student model on pseudo-labeled data."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STAGE 2: Pretraining Student Model (Iteration {iteration + 1})")
        logger.info("=" * 80)

        pretrain_data_path = self._get_pretrain_data_path(iteration)
        save_dir = str(self.results_dir / 'models' / f'pretrain_iter_{iteration:02d}')
        os.makedirs(save_dir, exist_ok=True)

        if not os.path.exists(pretrain_data_path):
            raise FileNotFoundError(f"Pretrain data not found: {pretrain_data_path}")

        cmd = [
            sys.executable, str(self.project_dir / 'train.py'),
            '--datadir', pretrain_data_path,
            '--pretrain', 'True',
            '--save-dir', save_dir,
        ] + self._get_train_args()

        result = self._run_command(cmd, f"Pretraining (iter {iteration + 1})")

        best_ckpt = self._parse_best_checkpoint(result.stdout, save_dir)
        self.checkpoint_history[f'pretrain_{iteration}'] = best_ckpt
        logger.info(f"Pretrain checkpoint: {best_ckpt}")
        return best_ckpt

    # ========================== STAGE 3 ==========================
    def finetune_student_model(self, iteration: int, pretrain_ckpt: Optional[str] = None) -> Optional[str]:
        """Stage 3: Finetune student model on high-quality DFT data."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STAGE 3: Finetuning Student Model (Iteration {iteration + 1})")
        logger.info("=" * 80)

        finetune_data_path = self.config['finetune_data']
        save_dir = str(self.results_dir / 'models' / f'finetune_iter_{iteration:02d}')
        os.makedirs(save_dir, exist_ok=True)

        if not os.path.exists(finetune_data_path):
            raise FileNotFoundError(f"Finetune data not found: {finetune_data_path}")

        cmd = [
            sys.executable, str(self.project_dir / 'train.py'),
            '--datadir', finetune_data_path,
            '--pretrain', 'False',
            '--save-dir', save_dir,
        ]
        if pretrain_ckpt and os.path.exists(pretrain_ckpt):
            cmd.extend(['--load-ckpt', pretrain_ckpt])
            logger.info(f"Loading pretrained weights from: {pretrain_ckpt}")
        else:
            logger.warning("No pretrained checkpoint provided; finetuning from scratch")

        cmd += self._get_train_args()

        result = self._run_command(cmd, f"Finetuning (iter {iteration + 1})")

        best_ckpt = self._parse_best_checkpoint(result.stdout, save_dir)
        self.checkpoint_history[f'finetune_{iteration}'] = best_ckpt
        logger.info(f"Finetune checkpoint: {best_ckpt}")
        return best_ckpt

    # ========================== STAGE 4 ==========================
    def run_md_simulation(self, iteration: int, finetune_ckpt: Optional[str] = None) -> str:
        """Stage 4: Run MD simulations to collect OOD configurations."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STAGE 4: Running MD Simulations (Iteration {iteration + 1})")
        logger.info("=" * 80)

        traj_dir = self.results_dir / 'md_trajectories' / f'iteration_{iteration:02d}'
        traj_dir.mkdir(parents=True, exist_ok=True)
        md_data_path = str(traj_dir / 'md_collected.pt')

        num_seeds = self.config.get('num_md_seeds_per_iter', 5)

        if self.test_mode:
            logger.info("Test mode: generating synthetic MD data")
            md_data = generate_synthetic_data(
                num_samples=max(3, num_seeds), seed=100 + iteration
            )
            torch.save(md_data, md_data_path)
            logger.info(f"Saved {len(md_data)} synthetic MD samples to {md_data_path}")
            return md_data_path

        logger.info(f"Running {num_seeds} MD simulations with finetuned student model...")
        for seed in range(num_seeds):
            logger.info(f"  MD simulation {seed + 1}/{num_seeds} (seed={seed})")
            cmd = [sys.executable, str(self.project_dir / 'Supp_traj_md.py'), '--seed', str(seed)]
            try:
                self._run_command(cmd, f"MD simulation seed={seed}")
            except RuntimeError:
                logger.warning(f"  MD simulation {seed} failed, continuing...")

        logger.info(f"MD simulations completed. Trajectories saved to {traj_dir}")
        return md_data_path

    # ========================== STAGE 5 ==========================
    def relabel_data(self, iteration: int, md_data_path: str) -> Dict[str, Any]:
        """
        Stage 5: Relabel and update pretraining data.

        Cross-relabeling strategy:
        - Retain a fraction of old pretrain data
        - Add new OOD data from MD simulations
        - Save combined dataset as the next iteration's pretrain data
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"STAGE 5: Relabeling Data (Iteration {iteration + 1})")
        logger.info("=" * 80)

        retention_rate = self.config.get('data_retention_rate', 0.5)

        old_pretrain_path = self._get_pretrain_data_path(iteration)
        new_pretrain_path = self._get_pretrain_data_path(iteration + 1)

        old_data = []
        if os.path.exists(old_pretrain_path):
            old_data = torch.load(old_pretrain_path, weights_only=False)
            logger.info(f"Loaded {len(old_data)} samples from old pretrain data")
        else:
            logger.warning(f"Old pretrain data not found: {old_pretrain_path}")

        new_data = []
        if os.path.exists(md_data_path):
            new_data = torch.load(md_data_path, weights_only=False)
            logger.info(f"Loaded {len(new_data)} samples from MD trajectories")
        else:
            logger.warning(f"MD data not found: {md_data_path}")

        num_retain = int(len(old_data) * retention_rate)
        if num_retain > 0 and len(old_data) > 0:
            indices = np.random.permutation(len(old_data))[:num_retain]
            retained_data = [old_data[i] for i in indices]
        else:
            retained_data = []

        combined_data = retained_data + new_data
        if len(combined_data) == 0:
            logger.warning("No data after relabeling! Keeping old data as-is.")
            combined_data = old_data

        torch.save(combined_data, new_pretrain_path)
        logger.info(f"Saved {len(combined_data)} samples to {new_pretrain_path}")
        logger.info(f"  Retained from old: {len(retained_data)}, New from MD: {len(new_data)}")

        relabel_metrics = {
            'relabel_iteration': iteration,
            'relabel_timestamp': datetime.now().isoformat(),
            'old_data_count': len(old_data),
            'retained_count': len(retained_data),
            'new_data_count': len(new_data),
            'combined_count': len(combined_data),
            'retention_rate': retention_rate,
        }
        return relabel_metrics

    # ========================== STAGE 6 ==========================
    def evaluate_models(self, iteration: int, finetune_ckpt: Optional[str] = None) -> Dict[str, Any]:
        """Stage 6: Evaluate model performance."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STAGE 6: Evaluating Models (Iteration {iteration + 1})")
        logger.info("=" * 80)

        metrics = {
            'eval_iteration': iteration,
            'eval_timestamp': datetime.now().isoformat(),
            'energy_mae': None,
            'force_mae': None,
            'status': 'pending',
        }

        if self.test_mode:
            logger.info("Test mode: computing evaluation metrics directly")
            metrics.update(self._evaluate_with_model(iteration, finetune_ckpt))
            return metrics

        cmd = [
            sys.executable, str(self.project_dir / 'inference.py'),
        ]
        if finetune_ckpt and os.path.exists(finetune_ckpt):
            cmd.extend(['--model-path', finetune_ckpt])

        try:
            result = self._run_command(cmd, f"Evaluation (iter {iteration + 1})")
            parsed = self._parse_eval_metrics(result.stdout)
            metrics.update(parsed)
            metrics['status'] = 'completed'
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            metrics['status'] = 'failed'

        return metrics

    def _evaluate_with_model(self, iteration: int, ckpt_path: Optional[str]) -> Dict[str, Any]:
        """Evaluate the model directly (used in test mode or when inference.py is unavailable)."""
        result = {'status': 'completed', 'force_mae': None, 'energy_mae': None}

        if not ckpt_path or not os.path.exists(ckpt_path):
            logger.warning("No checkpoint for evaluation")
            result['status'] = 'skipped'
            return result

        try:
            from PaiNN import PainnModel
            from torch_geometric.loader import DataLoader

            device = torch.device('cpu')
            model = PainnModel()
            state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
            new_state_dict = {k.replace('potential.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval()

            finetune_data = torch.load(self.config['finetune_data'], weights_only=False)
            val_data = finetune_data[int(len(finetune_data) * 0.85):]
            if len(val_data) == 0:
                val_data = finetune_data[-1:]
            loader = DataLoader(val_data, batch_size=1, shuffle=False)

            force_maes, energy_maes = [], []
            for data in loader:
                data.pos.requires_grad_(True)
                pred_energy, pred_forces = model(data)
                fmae = torch.mean((pred_forces - data.force).abs()).item()
                emae = torch.mean((pred_energy.squeeze() - data.energy).abs()).item()
                force_maes.append(fmae)
                energy_maes.append(emae)

            result['force_mae'] = float(np.mean(force_maes)) if force_maes else None
            result['energy_mae'] = float(np.mean(energy_maes)) if energy_maes else None
            logger.info(f"  Force MAE: {result['force_mae']:.6f}" if result['force_mae'] else "  Force MAE: N/A")
            logger.info(f"  Energy MAE: {result['energy_mae']:.6f}" if result['energy_mae'] else "  Energy MAE: N/A")
        except Exception as e:
            logger.warning(f"Direct evaluation failed: {e}")
            result['status'] = 'failed'

        return result

    def _parse_best_checkpoint(self, stdout: str, save_dir: str) -> Optional[str]:
        """Parse BEST_CHECKPOINT from train.py stdout, fallback to searching save_dir."""
        match = re.search(r'BEST_CHECKPOINT=(.+)', stdout)
        if match:
            path = match.group(1).strip()
            if os.path.exists(path):
                return path
        return self._find_best_checkpoint(save_dir)

    def _parse_eval_metrics(self, stdout: str) -> Dict[str, Any]:
        """Parse evaluation metrics from inference.py stdout."""
        metrics = {}
        force_match = re.search(r'Self-training Mean MAE:\s*([\d.]+)', stdout)
        if force_match:
            metrics['force_mae'] = float(force_match.group(1))
        energy_match = re.search(r'Energy MAE:\s*([\d.]+)', stdout)
        if energy_match:
            metrics['energy_mae'] = float(energy_match.group(1))
        return metrics

    # ========================== CONVERGENCE ==========================
    def check_convergence(self, metrics_history: list) -> bool:
        """Check if model has converged."""
        if len(metrics_history) < 2:
            return False

        threshold = self.config.get('convergence_threshold', 0.01)

        prev_mae = metrics_history[-2].get('force_mae')
        curr_mae = metrics_history[-1].get('force_mae')

        if prev_mae is None or curr_mae is None:
            logger.info("Convergence check: force_mae unavailable, skipping")
            return False

        if prev_mae == 0:
            return False

        improvement = (prev_mae - curr_mae) / abs(prev_mae)
        logger.info(f"Convergence check: improvement = {improvement:.4f} (threshold = {threshold:.4f})")
        return improvement < threshold

    # ========================== RESULTS ==========================
    def save_iteration_results(self, iteration: int, metrics: Dict[str, Any]):
        metrics_dir = self.results_dir / 'metrics'
        metrics_file = metrics_dir / f'iteration_{iteration:02d}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Iteration {iteration + 1} results saved to {metrics_file}")

    # ========================== MAIN PIPELINE ==========================
    def run_full_pipeline(self):
        logger.info("\n" + "=" * 80)
        logger.info("IPIP PIPELINE - STARTING FULL WORKFLOW")
        logger.info("=" * 80)
        logger.info(f"Total iterations: {self.config['num_iterations']}")
        logger.info(f"Start time: {datetime.now().isoformat()}")

        all_metrics = []

        try:
            # Stage 1: Generate initial pretraining data (only once)
            self.generate_initial_pretrain_data()

            # Also prepare finetune data for test mode
            if self.test_mode:
                finetune_path = self.config['finetune_data']
                if not os.path.exists(finetune_path):
                    finetune_data = generate_synthetic_data(num_samples=10, seed=99)
                    os.makedirs(os.path.dirname(finetune_path) or '.', exist_ok=True)
                    torch.save(finetune_data, finetune_path)
                    logger.info(f"Test mode: generated finetune data at {finetune_path}")

            for iter_num in range(self.config['num_iterations']):
                self.iteration = iter_num
                logger.info(f"\n{'#' * 80}")
                logger.info(f"# ITERATION {iter_num + 1} / {self.config['num_iterations']}")
                logger.info(f"{'#' * 80}")

                # Stage 2: Pretrain
                pretrain_ckpt = self.pretrain_student_model(iter_num)

                # Stage 3: Finetune (using pretrained checkpoint)
                finetune_ckpt = self.finetune_student_model(iter_num, pretrain_ckpt=pretrain_ckpt)

                # Stage 4: MD Simulation
                md_data_path = self.run_md_simulation(iter_num, finetune_ckpt=finetune_ckpt)

                # Stage 5: Relabel data (prepares data for next iteration)
                relabel_metrics = self.relabel_data(iter_num, md_data_path)

                # Stage 6: Evaluate
                eval_metrics = self.evaluate_models(iter_num, finetune_ckpt=finetune_ckpt)

                iteration_metrics = {
                    'iteration': iter_num,
                    'timestamp': datetime.now().isoformat(),
                    'pretrain_ckpt': pretrain_ckpt,
                    'finetune_ckpt': finetune_ckpt,
                    **{f'eval_{k}' if k in relabel_metrics else k: v for k, v in eval_metrics.items()},
                    **{f'relabel_{k}' if k in eval_metrics else k: v for k, v in relabel_metrics.items()},
                }
                all_metrics.append(iteration_metrics)
                self.save_iteration_results(iter_num, iteration_metrics)

                if self.check_convergence(all_metrics):
                    logger.info(f"\nModel converged at iteration {iter_num + 1}")
                    break

                logger.info(f"Iteration {iter_num + 1} completed")

            self.print_summary(all_metrics)

        except Exception as e:
            logger.error(f"\nIPIP pipeline failed: {e}", exc_info=True)
            raise

    def print_summary(self, metrics_history: list):
        logger.info("\n" + "=" * 80)
        logger.info("IPIP PIPELINE - FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total iterations completed: {len(metrics_history)}")
        logger.info(f"Results directory: {self.results_dir}")

        for metrics in metrics_history:
            i = metrics.get('iteration', '?')
            logger.info(f"\n  Iteration {i + 1}:")
            if metrics.get('energy_mae') is not None:
                logger.info(f"    Energy MAE: {metrics['energy_mae']:.6f}")
            if metrics.get('force_mae') is not None:
                logger.info(f"    Force MAE:  {metrics['force_mae']:.6f}")
            logger.info(f"    Status: {metrics.get('status', 'unknown')}")

        logger.info("\n" + "=" * 80)
        logger.info("IPIP pipeline completed successfully!")
        logger.info("=" * 80)


def create_default_config() -> Dict[str, Any]:
    return {
        'num_iterations': 3,
        'num_md_seeds': 10,
        'num_md_seeds_per_iter': 5,
        'finetune_data': './datasets/finetune_data.pt',
        'data_retention_rate': 0.5,
        'convergence_threshold': 0.01,
        'results_dir': './ipip_results',
        'project_dir': '.',
        'test_mode': False,
        'test_num_samples': 10,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run IPIP (Iterative Pretraining for Interatomic Potentials) pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 3 iterations with default settings
  python run_ipip_pipeline.py --iterations 3

  # Run quick test with synthetic data (no GPU needed)
  python run_ipip_pipeline.py --iterations 2 --test-mode

  # Use custom finetune data
  python run_ipip_pipeline.py --iterations 3 --finetune-data ./my_dft_data.pt
        """
    )

    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of IPIP iterations (default: 3)')
    parser.add_argument('--results-dir', type=str, default='./ipip_results',
                        help='Directory to save results')
    parser.add_argument('--project-dir', type=str, default='.',
                        help='Project root directory containing train.py etc.')
    parser.add_argument('--finetune-data', type=str, default='./datasets/finetune_data.pt',
                        help='Path to finetuning dataset (DFT data)')
    parser.add_argument('--pretrain-seeds', type=int, default=10,
                        help='Number of MD seeds for initial pretraining data generation')
    parser.add_argument('--md-seeds-per-iter', type=int, default=5,
                        help='Number of MD seeds per iteration')
    parser.add_argument('--convergence-threshold', type=float, default=0.01,
                        help='Relative improvement threshold for convergence')
    parser.add_argument('--data-retention-rate', type=float, default=0.5,
                        help='Proportion of old data to retain')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with synthetic data (no GPU needed)')
    parser.add_argument('--test-num-samples', type=int, default=10,
                        help='Number of synthetic samples in test mode')

    args = parser.parse_args()

    config = create_default_config()
    config.update({
        'num_iterations': args.iterations,
        'results_dir': args.results_dir,
        'project_dir': args.project_dir,
        'finetune_data': str(Path(args.finetune_data).resolve()),
        'num_md_seeds': args.pretrain_seeds,
        'num_md_seeds_per_iter': args.md_seeds_per_iter,
        'convergence_threshold': args.convergence_threshold,
        'data_retention_rate': args.data_retention_rate,
        'test_mode': args.test_mode,
        'test_num_samples': args.test_num_samples,
    })

    logger.info("IPIP Pipeline Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    pipeline = IPIPPipeline(config)
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()
