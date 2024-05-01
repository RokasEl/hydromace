from typing import List

import numpy as np
import torch
from ase import Atoms
from mace import data
from mace.calculators import MACECalculator
from mace.tools import AtomicNumberTable, torch_geometric

from .tools import _energies_to_real


def create_atomic_data_by_copy(
    configs: List[data.Configuration], z_table: AtomicNumberTable, cutoff: float
):
    """
    For Hessian calculation, all samples should have the same neighbour list.
    We can thus make the creation of atomic data faster by creating one AtomicData object properly,
    and the rest by copying the first and changing the positions.
    """
    atomic_data = data.AtomicData.from_config(configs[0], z_table, cutoff)
    data_dict = atomic_data.to_dict()
    dataset = [atomic_data]
    optional_keys = [
        "weight",
        "energy_weight",
        "forces_weight",
        "stress_weight",
        "virials_weight",
        "forces",
        "energy",
        "stress",
        "virials",
        "dipole",
        "charges",
    ]
    for config in configs[1:]:
        _data = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                _data[key] = value.clone()
            else:
                _data[key] = None
        _data["positions"] = torch.Tensor(config.positions)
        for key in optional_keys:
            if key not in _data:
                _data[key] = None
        dataset.append(data.AtomicData(**_data))
    return dataset


def calculate_mace_forces(
    calc: MACECalculator, samples: List[Atoms], batch_size: int = 32
):
    r_max = calc.r_max
    z_table = calc.z_table
    configs = [data.config_from_atoms(a) for a in samples]
    dataset = create_atomic_data_by_copy(configs, z_table, r_max)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,  # type:ignore
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    forces = []
    for batch in data_loader:
        batch = batch.to(calc.device)
        out = calc.models[0](batch)
        forces.append(out["forces"].detach())
    return torch.cat(forces, dim=0)


def normalise_and_fix_evec_shape(evecs: np.ndarray) -> np.ndarray:
    n_free = len(evecs)
    norm = np.linalg.norm(evecs.T, axis=0)
    evecs = np.divide(evecs.T, norm).T
    evecs = evecs.reshape(n_free, int(n_free / 3), 3)
    return evecs


def calculate_hessian_no_data_transfer(
    atoms: Atoms, calc: MACECalculator, delta: float = 0.01, batch_size: int = 32
):
    # Prepare the perturbed atoms
    perturbed_atoms = []
    for i in range(len(atoms)):
        for j in range(3):
            perturbed_plus = atoms.copy()
            perturbed_minus = atoms.copy()
            perturbed_plus.positions[i, j] += delta
            perturbed_minus.positions[i, j] -= delta
            perturbed_atoms.extend([perturbed_plus, perturbed_minus])

    forces = calculate_mace_forces(
        calc, perturbed_atoms, batch_size=batch_size
    )  # [2*len(atoms), len(atoms), 3]
    with torch.no_grad():
        hessian = torch.zeros(
            (3 * len(atoms), 3 * len(atoms)), device=calc.device, dtype=forces.dtype
        )
        forces = forces.view(-1, 2, len(atoms), 3)
        gradients = -(forces[:, 0] - forces[:, 1]) / (
            2 * delta
        )  # [len(atoms), len(atoms), 3]
        for i in range(len(atoms)):
            for j in range(3):
                hessian[:, i * 3 + j] = gradients[:, i, j]
    return hessian.cpu().numpy()


def calculate_evals_and_evecs(atoms, calc, delta=0.01, batch_size=32):
    hessian = calculate_hessian_no_data_transfer(
        atoms, calc, delta=delta, batch_size=batch_size
    )
    inverse_m = np.repeat(atoms.get_masses() ** -0.5, 3)
    e_vals, e_vecs = np.linalg.eigh(inverse_m[:, None] * hessian * inverse_m)
    e_vecs = e_vecs.T
    e_vals = _energies_to_real(e_vals)
    e_vecs = normalise_and_fix_evec_shape(e_vecs)
    return e_vals, e_vecs
