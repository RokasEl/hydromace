from typing import List

import ase
import ase.io as aio
import fire
from mace.data.utils import random_train_valid_split

from hydromace.tools import assign_num_hydrogens, remove_elements


def main(data_path: str, save_path: str, valid_fraction: float = 0.1, seed: int = 0):
    all_configs: List[ase.Atoms] = aio.read(data_path, index=":", format="extxyz")  # type: ignore
    prepared_configs = []
    for config in all_configs:
        if all(config.get_atomic_numbers() == 1):
            continue
        num_hydrogens = assign_num_hydrogens(config)
        config = remove_elements(config, [1])
        config.arrays["num_hydrogens"] = num_hydrogens
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
