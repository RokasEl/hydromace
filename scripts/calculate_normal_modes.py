import logging
import pathlib
from time import perf_counter

import ase.io as aio
import numpy as np
import torch
import typer
from ase.vibrations import Vibrations
from mace.calculators import mace_off

from hydromace.tools import (
    setup_logger,
    write_vibration_information_to_atoms,
)

app = typer.Typer()

setup_logger()


@app.command()
def main(
    data_path: str,
    save_path: str,
    index: str = ":",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    non_h_only: bool = False,
):
    data = aio.read(data_path, index=index, format="extxyz")
    calc = mace_off("medium", device=device, default_dtype="float64")
    start = perf_counter()
    logging.info(
        f"Starting normal mode calculations for indices {index} in {data_path}"
    )
    logging.info("Cleaning cache...")
    path = pathlib.Path("./vib/")
    if path.exists():
        for file in path.iterdir():
            file.unlink()
    for i, atoms in enumerate(data):
        atoms = atoms.copy()
        atoms.calc = calc
        if non_h_only:
            indices = np.where(atoms.get_atomic_numbers() != 1)[0]
        else:
            indices = None
        vibrations = Vibrations(atoms, indices=indices)
        vibrations.run()
        write_vibration_information_to_atoms(atoms, vibrations, non_h_only=non_h_only)
        aio.write(save_path, atoms, append=True, format="extxyz")
        vibrations.clean()
    duration = perf_counter() - start
    logging.info(f"Normal mode calculations completed in {duration:.2f} seconds.")


if __name__ == "__main__":
    app()
