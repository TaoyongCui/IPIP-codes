import torch
from torch_geometric.data import Data, DataLoader
import logging
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from painn import PainnModel

log = logging.getLogger(__name__)

__all__ = ["SchNetPackCalculator"]  # noqa: F822


class MLCalculator(Calculator):
    implemented_properties = ["energy", "forces", "uncertainty", "e_uncertainty"]

    def __init__(self, model_path, atoms=None, **kwargs):
        super().__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PainnModel(
            num_interactions=6, hidden_state_size=128, cutoff=6.0, pdb=True
        )
        state_dict = torch.load(model_path)["state_dict"]

        new_state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.atoms = atoms
        self.cutoff = 6.0
        self.model.to(self.device)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        inputs = self._prepare(atoms)
        inputs = inputs.to(self.device)
        energy, forces = self.model(inputs)
        # print(model_results)
        self.results["forces"] = forces.detach().cpu().numpy()
        self.results["energy"] = energy.cpu().item()

    def _prepare(self, atoms):
        """transform ASE atoms object to Data"""
        natom = len(atoms)
        cell = atoms.cell
        cell = torch.as_tensor(cell[:])
        pos = torch.as_tensor(atoms.positions, dtype=torch.float)
        x = torch.as_tensor(atoms.numbers, dtype=torch.int)
        # 受力从hartree/bohr转换为meV/A
        data = Data(
            pos=pos,
            z=x,
            batch=torch.zeros(natom, dtype=torch.long),
            natoms=torch.as_tensor([natom], dtype=torch.long),
            cell=cell,
            pbc=torch.as_tensor(atoms.pbc, dtype=torch.bool),
        )
        data.to(self.device)
        return data
