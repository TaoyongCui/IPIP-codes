#!/usr/bin/env python3
"""
End-to-end test for the IPIP pipeline using 10 synthetic data points.
Runs entirely on CPU with minimal training to verify pipeline logic.

Usage:
    python test_pipeline.py
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

TEST_DIR = Path('./ipip_test_results')


def clean_test_dir():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline_test():
    project_dir = Path(__file__).parent.resolve()

    print("=" * 70)
    print("IPIP PIPELINE - END-TO-END TEST (10 synthetic samples)")
    print("=" * 70)
    print(f"Project directory: {project_dir}")
    print(f"Test results directory: {TEST_DIR.resolve()}")
    print()

    clean_test_dir()

    finetune_path = str((TEST_DIR / 'finetune_test.pt').resolve())

    cmd = [
        sys.executable, str(project_dir / 'run_ipip_pipeline.py'),
        '--iterations', '2',
        '--results-dir', str(TEST_DIR.resolve()),
        '--project-dir', str(project_dir),
        '--finetune-data', finetune_path,
        '--pretrain-seeds', '2',
        '--md-seeds-per-iter', '2',
        '--convergence-threshold', '0.001',
        '--data-retention-rate', '0.5',
        '--test-mode',
        '--test-num-samples', '10',
    ]

    print("Running command:")
    print(f"  {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(project_dir))

    print()
    print("=" * 70)
    if result.returncode == 0:
        print("TEST PASSED - Pipeline completed successfully!")
        print_test_artifacts()
    else:
        print(f"TEST FAILED - Exit code: {result.returncode}")
    print("=" * 70)

    return result.returncode


def print_test_artifacts():
    """List all files produced by the test run."""
    print("\nGenerated artifacts:")
    for root, dirs, files in os.walk(TEST_DIR):
        level = len(Path(root).relative_to(TEST_DIR).parts)
        indent = "  " * level
        print(f"{indent}{Path(root).name}/")
        for f in sorted(files):
            fpath = Path(root) / f
            size_kb = fpath.stat().st_size / 1024
            print(f"{indent}  {f}  ({size_kb:.1f} KB)")


if __name__ == '__main__':
    sys.exit(run_pipeline_test())
