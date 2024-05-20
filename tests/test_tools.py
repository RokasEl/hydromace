import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from e3nn import o3
from mace.modules.blocks import RealAgnosticInteractionBlock
from mace.tools import AtomicNumberTable

from hydromace.interface import HydroMACE
from hydromace.tools import (
    assign_num_hydrogens_and_parent_heavy_atoms,
    get_model_dtype,
    remove_selected_hydrogens_from_batch,
    sample_hydrogens_to_remove,
)
from hydromace.training_tools import get_dataloaders


@pytest.fixture
def batch_of_data():
    atoms = [
        molecule("C6H6"),
        molecule("N2"),
        molecule("CH3CH2OH"),
    ]
    for mol in atoms:
        num_hs, parent_heavy_atom = assign_num_hydrogens_and_parent_heavy_atoms(mol)
        mol.arrays["charges"] = num_hs
        parent_atoms = np.zeros((len(mol), 3))
        parent_atoms[:, 0] = parent_heavy_atom
        mol.arrays["forces"] = (
            parent_atoms  # hack to add parent heavy atom to atoms object
        )

    dataloader, _ = get_dataloaders(
        atoms,
        batch_size=3,
        z_table=AtomicNumberTable([1, 6, 7, 8]),
        hydrogen_number_key="charges",
        cutoff=3.5,
        shuffle=False,
    )
    return next(iter(dataloader))


@pytest.fixture
def hydromace_model():
    return HydroMACE(
        atomic_inter_scale=0.0,
        atomic_inter_shift=0.0,
        r_max=3.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=3,
        interaction_cls=RealAgnosticInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=2,
        num_elements=4,
        hidden_irreps=o3.Irreps("16x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.array([0.0, 0.0, 0.0, 0.0]),
        avg_num_neighbors=4,
        atomic_numbers=[1, 6, 7, 8],
        correlation=3,
        gate=None,
    )


@pytest.mark.parametrize(
    "atoms, expected_num_h, expected_parent_heavy_atom",
    [
        (
            molecule("C6H6"),
            np.array([1] * 6 + [0] * 6),
            np.array([-1] * 6 + list(range(6))),
        ),
        (molecule("N2"), np.array([0] * 2), np.array([-1] * 2)),
        (
            molecule("CH3CH2OH"),
            np.array([3, 2, 1] + [0] * 6),
            np.array([-1] * 3 + [2, 1, 1] + [0] * 3),
        ),
        (
            molecule("CH3CH2OH")[::-1],
            np.array([0] * 6 + [1, 2, 3]),
            np.array([8] * 3 + [7, 7, 6] + [-1] * 3),
        ),
        (
            Atoms(
                "HCHCHHHH",
                positions=[
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                    [2, 0, 0],
                    [0, 0, 0],
                    [2, 0, 0],
                    [2, 0, 0],
                    [2, 0, 0],
                ],
            ),
            np.array([0, 3, 0, 3, 0, 0, 0, 0]),
            np.array([1, -1, 1, -1, 1, 3, 3, 3]),
        ),
    ],
)
def test_assign_num_hydrogens(atoms, expected_num_h, expected_parent_heavy_atom):
    num_h, parent_heavy_atoms = assign_num_hydrogens_and_parent_heavy_atoms(atoms)
    assert np.allclose(num_h, expected_num_h)
    assert np.allclose(parent_heavy_atoms, expected_parent_heavy_atom)


def test_assigning_model_dtype():
    model = torch.nn.Linear(5, 7)
    model.to(torch.float64)
    assert get_model_dtype(model) == torch.float64

    model.to(torch.float32)
    assert get_model_dtype(model) == torch.float32


def test_sample_hydrogens_to_remove(batch_of_data):
    assert sum(batch_of_data.charges) == 12
    to_remove = sample_hydrogens_to_remove(batch_of_data)
    assert to_remove.shape == (len(batch_of_data.batch),)
    subgraphs = torch.unique(batch_of_data.batch)

    for subgraph in subgraphs:
        num_hs_in_subgraph = torch.sum(
            batch_of_data.charges[batch_of_data.batch == subgraph]
        )
        assert sum(to_remove[batch_of_data.batch == subgraph]) <= num_hs_in_subgraph

    # Check only hydrogens are flagged for removal
    selected_indices = torch.where(to_remove == 1)[0]
    selected_elements = batch_of_data.node_attrs[selected_indices]
    assert torch.all(selected_elements == torch.Tensor([1, 0, 0, 0]))


def test_sample_hydrogens_to_remove_removes_all_hs_correct_fraction_of_time(
    batch_of_data,
):

    to_remove = sample_hydrogens_to_remove(batch_of_data, full_removal_frequency=1.0)
    subgraphs = torch.unique(batch_of_data.batch)
    for subgraph in subgraphs:
        num_hs_in_subgraph = torch.sum(
            batch_of_data.charges[batch_of_data.batch == subgraph]
        )
        assert sum(to_remove[batch_of_data.batch == subgraph]) == num_hs_in_subgraph

    # Check for random removal that all hydrogens are removed equally often
    to_remove_not_full = [
        sample_hydrogens_to_remove(batch_of_data, full_removal_frequency=0.0)
        for _ in range(500)
    ]
    to_remove_not_full = torch.stack(to_remove_not_full, dim=0).sum(dim=0)
    to_remove_not_full = to_remove_not_full / to_remove_not_full.sum()
    assert torch.allclose(
        to_remove_not_full,
        torch.ones_like(to_remove_not_full) / len(to_remove_not_full),
        atol=0.05,
    )


def test_remove_selected_hydrogens_from_the_batch(batch_of_data):

    to_remove = sample_hydrogens_to_remove(batch_of_data, full_removal_frequency=1.0)
    adjusted_batch = remove_selected_hydrogens_from_batch(batch_of_data, to_remove)
    # no hydrogens should be left
    assert torch.all(adjusted_batch.node_attrs[:, 0] == 0)
    expected_missing_hs = torch.tensor([1] * 6 + [0] * 2 + [3, 2, 1])
    assert torch.all(adjusted_batch.charges == expected_missing_hs)

    to_remove = torch.zeros_like(to_remove)
    to_remove[7] = 1
    adjusted_batch = remove_selected_hydrogens_from_batch(batch_of_data, to_remove)
    assert len(adjusted_batch.batch) == len(batch_of_data.batch) - 1
    expected_missing_hs = torch.zeros_like(adjusted_batch.charges)
    expected_missing_hs[batch_of_data.forces[7, 0].to(torch.long)] = 1
    assert torch.all(adjusted_batch.charges == expected_missing_hs)


def test_model_produces_output_with_adjusted_batch(batch_of_data, hydromace_model):

    to_remove = sample_hydrogens_to_remove(batch_of_data, full_removal_frequency=1.0)
    adjusted_batch = remove_selected_hydrogens_from_batch(batch_of_data, to_remove)
    _ = hydromace_model(adjusted_batch)


def test_backward_pass_with_adjusted_batch(batch_of_data, hydromace_model):

    to_remove = sample_hydrogens_to_remove(batch_of_data, full_removal_frequency=1.0)
    adjusted_batch = remove_selected_hydrogens_from_batch(batch_of_data, to_remove)
    output = hydromace_model(adjusted_batch)
    predicted_missing_hydrogens = output["missing_hydrogen_logits"]
    actual_missing_hydrogens = adjusted_batch["charges"].to(torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(predicted_missing_hydrogens, actual_missing_hydrogens)
    loss.backward()
