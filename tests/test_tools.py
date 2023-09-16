import numpy as np
import pytest
import torch
from ase.build import molecule

from hydromace.tools import assign_num_hydrogens, get_model_dtype


@pytest.mark.parametrize(
    "atoms, expected",
    [
        (molecule("C6H6"), np.array([1] * 6)),
        (molecule("N2"), np.array([0] * 2)),
        (molecule("CH3CH2OH"), np.array([3, 2, 1])),
    ],
)
def test_assign_num_hydrogens(atoms, expected):
    out = assign_num_hydrogens(atoms)
    assert np.allclose(out, expected)


def test_assigning_model_dtype():
    model = torch.nn.Linear(5, 7)
    model.to(torch.float64)
    assert get_model_dtype(model) == torch.float64

    model.to(torch.float32)
    assert get_model_dtype(model) == torch.float32
