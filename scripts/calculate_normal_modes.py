from time import perf_counter

import ase.io as aio
import torch
import typer
from ase.vibrations import Vibrations
from mace.calculators import mace_off

from hydromace.tools import write_vibration_information_to_atoms

app = typer.Typer()


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
    print(f"Starting normal mode calculations for indices {index} in {data_path}")
    for atoms in data:
        atoms.calc = calc
        vibrations = Vibrations(atoms)
        vibrations.run()
        write_vibration_information_to_atoms(atoms, vibrations)
        aio.write(save_path, atoms, append=True, format="extxyz")
    duration = perf_counter() - start
    print(f"Normal mode calculations completed in {duration:.2f} seconds.")


if __name__ == "__main__":
    app()
