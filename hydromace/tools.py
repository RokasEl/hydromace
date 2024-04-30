from typing import List

import numpy as np
import torch
from ase import Atoms
from ase.vibrations import Vibrations


def assign_num_hydrogens(atoms: Atoms) -> np.ndarray:
    postions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    is_hydrogen = atomic_numbers == 1
    if np.sum(is_hydrogen) == 0:
        return np.zeros(len(atoms), dtype=int)
    hydrogen_positions = postions[is_hydrogen]
    heavy_atom_positions = postions[~is_hydrogen]
    distances = np.linalg.norm(
        hydrogen_positions[:, None, :] - heavy_atom_positions[None, :, :], axis=-1
    )
    closest_heavy_atoms = np.argmin(distances, axis=-1)
    atoms_with_hs, num_hs = np.unique(closest_heavy_atoms, return_counts=True)
    num_hydrogens = np.zeros(sum(~is_hydrogen), dtype=int)
    for idx, num in zip(atoms_with_hs, num_hs):
        num_hydrogens[idx] = num
    return num_hydrogens


def _energies_to_real(energies: np.ndarray) -> np.ndarray:
    """
    Ensure all values in the array are real. Use negative values to indicate imaginary components.
    """
    energies_real = np.zeros(energies.shape, dtype=float)
    for i, energy in enumerate(energies):
        if energy.imag == 0:
            energies_real[i] = energy.real
        elif energy.real == 0 and energy.imag != 0:
            energies_real[i] = -energy.imag
        else:
            raise ValueError("Energy has both real and imaginary components")
    return energies_real


def write_vibration_information_to_atoms(atoms: Atoms, vibrations: Vibrations) -> None:
    """
    Write the vibration information to the atoms object
    """
    vibration_data = vibrations.get_vibrations()
    energies, modes = vibration_data.get_energies_and_modes()
    energies = _energies_to_real(energies)
    atoms.info["vibration_energy"] = energies
    for i in range(modes.shape[0]):
        atoms.arrays[f"vibration_mode_{i}"] = modes[i]


# From moldiff package.
def remove_elements(atoms: Atoms, atomic_numbers_to_remove: List[int]) -> Atoms:
    """
    Remove all hydrogens from the atoms object
    """
    atoms_copy = atoms.copy()
    for atomic_number in atomic_numbers_to_remove:
        to_remove = atoms_copy.get_atomic_numbers() == atomic_number
        del atoms_copy[to_remove]
    return atoms_copy


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    dtypes = set()
    for p in model.parameters():
        dtypes.add(p.dtype)
    if torch.float32 in dtypes:
        return torch.float32
    elif torch.float64 in dtypes:
        return torch.float64
    else:
        raise ValueError("Model neither float32 or float64")
