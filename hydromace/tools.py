import logging
import os
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from mace.tools import torch_geometric


def assign_num_hydrogens_and_parent_heavy_atoms(
    atoms: Atoms,
) -> Tuple[np.ndarray, np.ndarray]:
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    is_hydrogen = atomic_numbers == 1
    if np.sum(is_hydrogen) == 0:
        return np.zeros(len(atoms), dtype=int), np.zeros(len(atoms), dtype=int) - 1
    hydrogen_positions = positions[is_hydrogen]
    heavy_atom_positions = positions[~is_hydrogen]
    distances = np.linalg.norm(
        hydrogen_positions[:, None, :] - heavy_atom_positions[None, :, :], axis=-1
    )
    closest_heavy_atoms = np.argmin(distances, axis=-1)
    atoms_with_hs, num_hs = np.unique(closest_heavy_atoms, return_counts=True)

    num_hydrogens = np.zeros(len(atoms), dtype=int)
    non_hydrogen_indices = np.arange(len(atoms))[~is_hydrogen]
    for i in range(len(atoms_with_hs)):
        num_hydrogens[non_hydrogen_indices[atoms_with_hs[i]]] = num_hs[i]

    parent_atoms = np.zeros(len(atoms), dtype=int) - 1
    hydrogen_indices = np.arange(len(atoms))[is_hydrogen]
    for i in range(len(hydrogen_indices)):
        parent_atoms[hydrogen_indices[i]] = non_hydrogen_indices[closest_heavy_atoms[i]]
    return num_hydrogens, parent_atoms


def sample_hydrogens_to_remove(
    batch: torch_geometric.data.Data, full_removal_frequency: float = 0.5
) -> torch.Tensor:
    subgraphs = torch.unique(batch.batch)
    full_removal = torch.rand(subgraphs.shape) < full_removal_frequency
    to_remove = torch.zeros(len(batch.batch), dtype=torch.long)

    for i, subgraph in enumerate(subgraphs):
        h_indices = torch.where(
            (batch.batch == subgraph) & (batch.node_attrs[:, 0] == 1)
        )[0]

        num_hs_in_subgraph = (
            torch.sum(batch.charges[batch.batch == subgraph]).to(torch.long).item()
        )
        if num_hs_in_subgraph == 0:
            continue

        if full_removal[i]:
            to_remove[h_indices] = 1
        else:
            num_hs_to_remove = torch.randint(0, num_hs_in_subgraph, (1,)).item()
            to_remove[h_indices[torch.randperm(len(h_indices))[:num_hs_to_remove]]] = 1
    return to_remove


def remove_selected_hydrogens_from_batch(
    batch_of_data: torch_geometric.data.Data, to_remove: torch.Tensor
):
    """
    Remove selected nodes and associated edges from the batch. Adjust the number of missing hydrogens.
    """
    batch = batch_of_data.clone()
    h_assignments = batch.forces[:, 0].to(torch.long)
    # h_assignments are relative to the batch, so we need to adjust them
    index_adjustment = torch.cumsum(torch.bincount(batch.batch), 0)
    for i in range(1, len(index_adjustment)):
        h_assignments[batch_of_data.batch == i] += index_adjustment[i - 1]
    h_assignments = h_assignments[to_remove == 1]
    missing_hs = torch.zeros_like(to_remove)
    missing_hs.scatter_reduce_(0, h_assignments, torch.ones_like(h_assignments), "sum")
    batch_of_data.charges = missing_hs

    to_keep = 1 - to_remove
    batch.positions = batch_of_data.positions[to_keep == 1]
    batch.node_attrs = batch_of_data.node_attrs[to_keep == 1]
    batch.charges = batch_of_data.charges[to_keep == 1]
    batch.batch = batch_of_data.batch[to_keep == 1]

    # Remove edges
    edge_indices = torch.where(
        to_keep[batch.edge_index[0]] * to_keep[batch.edge_index[1]]
    )[0]
    new_edge_index = batch.edge_index[:, edge_indices]
    batch.shifts = batch.shifts[edge_indices]
    batch.unit_shifts = batch.unit_shifts[edge_indices]
    # Reindex the edges
    edge_index_map = torch.zeros_like(to_keep)
    edge_index_map[to_keep == 1] = torch.arange(torch.sum(to_keep))
    new_edge_index = edge_index_map[new_edge_index]
    batch.edge_index = new_edge_index

    return batch


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


def write_vibration_information_to_atoms(
    atoms: Atoms, evals: np.ndarray, evecs: np.ndarray
) -> None:
    atoms.info["vibration_energy"] = evals
    for i in range(evecs.shape[0]):
        atoms.arrays[f"vibration_mode_{i}"] = evecs[i]


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


# Taken from MACE
def setup_logger(
    name: str | None = None,
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + ".log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)
