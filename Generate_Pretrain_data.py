import argparse
import os
from pathlib import Path
import numpy as np
from ase import io, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# --- Import your custom calculator ---
# Ensure that PAINN_Calculator.py is in the same directory or in your PYTHONPATH
from mace.calculators import mace_off


def run_ase_simulation(seed, n_steps=50000, temp=340.0, timestep=1.0, traj_interval=50):
    """
    Runs a molecular dynamics simulation using ASE.

    Args:
        seed (int): The random seed for velocity initialization.
        n_steps (int): Total number of simulation steps.
        temp (float): The target temperature in Kelvin.
        timestep (float): The integration timestep in femtoseconds.
        traj_interval (int): How often to write trajectory frames.
    """
    print(f"Starting ASE simulation with seed={seed}.")

    # --- 1. Setup the Simulation Environment ---
    # Create a directory to store the output trajectory
    output_dir = Path("./MACE-chignolin")
    logs_dir = Path("./ase_logs/")
    logs_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    trajectory_file = f"{output_dir}/traj_{seed}.traj"

    # --- 2. Read the Initial Structure ---
    # ASE can read LAMMPS data files directly
    atoms = io.read("init_structures.xyz", index=seed)
    atoms.center(vacuum=5.0)  # Center the structure in a larger box
    atoms.pbc = False
    # --- 3. Initialize Velocities ---
    # This corresponds to LAMMPS's 'velocity all create' command
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temp, rng=np.random.RandomState(seed)
    )
    # The following two lines are good practice to prevent the whole system from drifting or rotating

    # --- 4. Attach the Calculator ---
    calculator = mace_off(
        "MACE-OFF24_medium.model", device="cuda", default_dtype="float32"
    )  # Modify this line as needed
    atoms.calc = calculator

    # --- 5. Setup the MD Integrator ---
    # We use the Langevin integrator, which corresponds to NVT
    # The friction (or time-scale) parameter is analogous to LAMMPS's 't-damp'
    # ASE timestep is in femtoseconds
    dyn = Langevin(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temp,
        friction=0.01,
        logfile=str(logs_dir / f"ase_md_log_{seed}.txt"),
        loginterval=traj_interval,
    )

    # --- 6. Define How to Save the Trajectory ---
    # This replaces the 'dump' command in LAMMPS

    # Attach the logger and trajectory writer to the dynamics
    dyn.attach(
        io.Trajectory(str(trajectory_file), "w", atoms=atoms), interval=traj_interval
    )
    # dyn.attach(log_trajectory, interval=traj_interval)

    # --- 7. Run the Simulation ---
    print(f"Running NVT dynamics for {n_steps} steps...")
    dyn.run(n_steps)

    print(f"Simulation with seed={seed} completed successfully.")
    print(f"Trajectory saved to: {trajectory_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch MD simulations using ASE.")
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for velocity initialization.",
    )

    args = parser.parse_args()

    run_ase_simulation(seed=args.seed)
