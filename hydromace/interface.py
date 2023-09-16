from functools import partial
from typing import List

import ase
import numpy as np
import torch
from mace.data.atomic_data import AtomicData, get_data_loader
from mace.data.utils import config_from_atoms
from mace.tools import AtomicNumberTable

from .model import HydroMACE
from .tools import get_model_dtype


# From moldiff package.
def convert_atoms_to_atomic_data(
    atoms: ase.Atoms | List[ase.Atoms],
    z_table: AtomicNumberTable,
    cutoff: float,
    device: str,
):
    if isinstance(atoms, ase.Atoms):
        atoms = [atoms]
    confs = [config_from_atoms(x) for x in atoms]
    atomic_datas = [
        AtomicData.from_config(conf, z_table, cutoff).to(device) for conf in confs
    ]
    return atomic_datas


def batch_atoms(
    atoms: ase.Atoms | list[ase.Atoms],
    z_table: AtomicNumberTable,
    cutoff: float,
    device: str,
) -> AtomicData:
    atomic_datas = convert_atoms_to_atomic_data(atoms, z_table, cutoff, device)
    return next(
        iter(get_data_loader(atomic_datas, batch_size=len(atomic_datas), shuffle=False))
    )


class HydroMaceCalculator:
    def __init__(
        self,
        model: HydroMACE,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.dtype = get_model_dtype(model)
        self.z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])  # type: ignore
        self.cutoff = model.r_max.item()  # type: ignore
        self.batch_atoms = partial(
            batch_atoms, z_table=self.z_table, cutoff=self.cutoff, device=self.device
        )

    def predict_missing_hydrogens(self, atoms: ase.Atoms) -> np.ndarray:
        batched = self.batch_atoms(atoms)
        keys = filter(lambda x: torch.is_floating_point(batched[x]), batched.keys)  # type: ignore
        batched = batched.to(self.dtype, *keys)
        with torch.no_grad():
            outputs = self.model(batched)
        num_hydrogens = outputs["missing_hydrogens"].cpu().numpy()
        atoms.info["num_predicted_hydrogens"] = num_hydrogens
        return num_hydrogens
