import logging
import pathlib
from time import perf_counter

import ase.io as aio
import torch
import typer
from mace.calculators import mace_off

from hydromace.hessian_calculation import calculate_evals_and_evecs
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
):
    data = aio.read(data_path, index=index, format="extxyz")
    calc = mace_off("medium", device=device, default_dtype="float64")
    start = perf_counter()
    logging.info(
        f"Starting normal mode calculations for indices {index} in {data_path}"
    )
    save_path_obj = pathlib.Path(save_path)
    if save_path_obj.exists():
        logging.warning(f"Overwriting {save_path}")
        save_path_obj.unlink()
    for i, atoms in enumerate(data):
        atoms = atoms.copy()  # type:ignore
        atoms.calc = calc
        evals, evecs = calculate_evals_and_evecs(atoms, calc, batch_size=32)
        write_vibration_information_to_atoms(atoms, evals, evecs)
        aio.write(save_path, atoms, append=True, format="extxyz")
    duration = perf_counter() - start
    logging.info(f"Normal mode calculations completed in {duration:.2f} seconds.")


if __name__ == "__main__":
    app()
