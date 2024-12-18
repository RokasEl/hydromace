from typing import List

import ase
import ase.io as aio
import fire
import numpy as np
from mace.data.utils import random_train_valid_split

from hydromace.tools import (
    assign_num_hydrogens_and_parent_heavy_atoms,
    remove_elements,
)


def main(data_path: str, save_path: str, valid_fraction: float = 0.1, seed: int = 0):
    all_configs: List[ase.Atoms] = aio.read(data_path, index=":", format="extxyz")  # type: ignore
    prepared_configs = []
    for config in all_configs:
        config = config.copy()
        if all(config.get_atomic_numbers() == 1):
            continue
        num_hydrogens, parent_heavy_atoms = assign_num_hydrogens_and_parent_heavy_atoms(
            config
        )
        config.arrays["num_hydrogens"] = num_hydrogens
        parents_atoms = np.zeros((len(config), 3))
        parents_atoms[:, 0] = parent_heavy_atoms
        config.arrays["forces"] = parents_atoms
        prepared_configs.append(config)
    train_data, test_data = random_train_valid_split(
        prepared_configs, valid_fraction, seed
    )
    train_name = save_path.replace(".xyz", "_train.xyz")
    test_name = save_path.replace(".xyz", "_test.xyz")
    aio.write(train_name, train_data, format="extxyz")  # type: ignore
    aio.write(test_name, test_data, format="extxyz")  # type: ignore


if __name__ == "__main__":
    fire.Fire(main)
