import pathlib
from dataclasses import dataclass
from typing import List

import ase
import ase.io as aio
import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from hydromace.interface import HydroMaceCalculator
from hydromace.tools import get_model_dtype, sample_hydrogens_to_remove, remove_selected_hydrogens_from_batch
from hydromace.training_tools import get_dataloaders


@dataclass
class Results:
    model_path: str
    num_wrong_hydrogens: int
    total_num_hydrogens: int


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = typer.Typer()


def evaluate_model(model_path: str, dataloader: DataLoader):
    model = torch.load(model_path)
    model.eval()
    num_wrong_assignments = 0
    total_num_hs = 0
    num_preds = 0
    with torch.no_grad():
        for batch in dataloader:
            to_remove = sample_hydrogens_to_remove(batch, full_removal_frequency=1.)
            batch = remove_selected_hydrogens_from_batch(batch, to_remove)
            keys = filter(lambda x: torch.is_floating_point(batch[x]), batch.keys)
            batch = batch.to(get_model_dtype(model), *keys).to(DEVICE)
            out = model(batch)
            pred = out["missing_hydrogens"]
            true = batch.charges
            num_wrong_assignments += torch.sum(torch.abs(pred - true)).item()
            total_num_hs += torch.sum(true).item()
            num_preds += pred.shape[0]

    results = Results(
        model_path,
        int(num_wrong_assignments),
        int(total_num_hs),
    )
    return results


def initialize_dataloader(model_path: str, atoms: List[ase.Atoms], batch_size=32):
    model = torch.load(model_path, map_location=DEVICE)
    calc = HydroMaceCalculator(model)
    z_table = calc.z_table
    dataloader, _ = get_dataloaders(
        atoms=atoms,
        batch_size=batch_size,
        z_table=z_table,
        hydrogen_number_key="num_hydrogens",
        cutoff=calc.cutoff,
        valid_fraction=0,
        drop_last=False,
        shuffle=False,
    )
    return dataloader


@app.command()
def main(
    model_dir: str,
    data_path: str,
    save_path: str = "param_sweep_results.csv",
    batch_size: int = 32,
):
    print(f"Evaluating models in {model_dir} using data from {data_path}")
    atoms: List[ase.Atoms] = aio.read(data_path, index=":", format="extxyz")  # type: ignore
    all_results = []
    first = True
    for model_file in tqdm(pathlib.Path(model_dir).glob("*.model")):
        if first:
            first = False
            dataloader = initialize_dataloader(
                model_file.as_posix(), atoms, batch_size=batch_size
            )
        try:
            results = evaluate_model(model_file.as_posix(), dataloader)
        except Exception as e:
            print(f"Error evaluating {model_file}: {e}")
            continue
        all_results.append(results)
    df = pd.DataFrame(all_results)
    df.to_csv(save_path)


if __name__ == "__main__":
    app()
