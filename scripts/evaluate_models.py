import pathlib
from dataclasses import dataclass
from typing import List

import ase
import ase.io as aio
import numpy as np
import pandas as pd
import torch
import typer
from tqdm import tqdm

from hydromace.interface import HydroMaceCalculator


@dataclass
class Results:
    model_path: str
    num_wrong_hydrogens: int
    total_num_hydrogens: int
    full_asignment_rate: float


app = typer.Typer()


def evaluate_model(model_path: str, atoms: List[ase.Atoms]):
    model = torch.load(model_path)
    model.eval()
    calc = HydroMaceCalculator(model)
    num_wrong_assignments = 0
    total_num_hs = 0
    all_correct_assignments = 0
    for mol in atoms:
        pred = calc.predict_missing_hydrogens(mol)
        true = mol.arrays["num_hydrogens"]
        num_wrong_assignments += np.sum(np.abs(pred - true))
        total_num_hs += np.sum(true)
        if np.all(pred == true):
            all_correct_assignments += 1
    results = Results(
        model_path,
        num_wrong_assignments,
        total_num_hs,
        all_correct_assignments / len(atoms),
    )
    return results


@app.command()
def main(model_dir: str, data_path: str):
    print(f"Evaluating models in {model_dir} using data from {data_path}")
    atoms = aio.read(data_path, index=":", format="extxyz")
    all_results = []
    for model_file in tqdm(pathlib.Path(model_dir).glob("*.model")):
        results = evaluate_model(model_file.as_posix(), atoms)
        all_results.append(results)
    df = pd.DataFrame(all_results)
    print(df)


if __name__ == "__main__":
    app()
