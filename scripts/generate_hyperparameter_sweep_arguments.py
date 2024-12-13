from dataclasses import dataclass

import numpy as np
import typer

"""
Script to generate hyperparameter sweep arguments for the `train.py` script.
"""


def mlp_irreps_to_str(mlp_irreps: int):
    return f"{mlp_irreps}x0e"


@dataclass
class Trial:
    num_interactions: int = 1
    num_radial_basis: int = 4
    noise_scale: float = 0.5
    max_L: int = 2
    num_channels: int = 16
    r_max: float = 4.0
    num_cutoff_basis: int = 4
    mlp_irreps: str = mlp_irreps_to_str(16)

    def to_dict(self):
        return self.__dict__

    def __str__(self) -> str:
        d = self.__dict__
        return ";".join([f"{v}" for k, v in d.items()])


app = typer.Typer()


@app.command()
def main(save_path: str, num_trials: int = 100):
    # define the search space
    rng = np.random.default_rng(seed=0)
    for _ in range(num_trials):
        trial = Trial(
            num_interactions=rng.choice(
                [
                    1,
                    2,
                ]
            ),
            num_radial_basis=rng.choice([4, 6, 8]),
            noise_scale=np.round(rng.uniform(0.1, 0.5), 2),
            max_L=rng.choice(
                [
                    1,
                    2,
                ]
            ),
            num_channels=rng.choice([16, 32, 64, 96, 128]),
            r_max=np.round(rng.uniform(3.0, 5.0), 1),
            num_cutoff_basis=rng.choice([4, 6, 8]),
            mlp_irreps=mlp_irreps_to_str(rng.choice([16, 32, 64, 96, 128])),
        )
        with open(save_path, "a") as f:
            f.write(f"{str(trial)}\n")


if __name__ == "__main__":
    app()
