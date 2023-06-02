import ast
import json
import logging
from typing import List

import numpy as np
import torch
from ase import Atoms
from e3nn import o3
from mace import modules
from mace.data import AtomicData
from mace.data.utils import (
    config_from_atoms_list,
    random_train_valid_split,
)
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.dataloader import DataLoader

from .model import HydroMACE

"""
Functions to reduce bloat in the run_train.py script. Many of these functions are copied from the MACE package.
"""


def get_dataloaders(
    atoms: List[Atoms],
    batch_size: int,
    z_table: AtomicNumberTable,
    hydrogen_number_key: str,
    cutoff: float,
    valid_fraction: float = 0,
    seed: int = 0,
):
    configs = config_from_atoms_list(
        atoms, config_type_weights={"Default": 1.0}, charges_key=hydrogen_number_key
    )
    if 0.0 < valid_fraction < 1.0:
        train_configs, valid_configs = random_train_valid_split(
            configs, valid_fraction, seed
        )
    else:
        train_configs = configs
        valid_configs = None

    train_atomic_data = [
        AtomicData.from_config(x, z_table, cutoff) for x in train_configs
    ]
    train_dataloader = DataLoader(
        train_atomic_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    if valid_configs is not None:
        valid_atomic_data = [
            AtomicData.from_config(x, z_table, cutoff) for x in valid_configs
        ]
        valid_dataloader = DataLoader(
            valid_atomic_data, batch_size=batch_size, shuffle=False, drop_last=False
        )
    else:
        valid_dataloader = None
    return train_dataloader, valid_dataloader


def handle_e3nn_args(args):
    if args.num_channels is not None and args.max_L is not None:
        assert args.num_channels > 0, "num_channels must be positive integer"
        assert args.max_L >= 0, "max_L must be non-negative integer"
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )

    assert (
        len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
    ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"
    return args


def parse_args_into_model_config(args, z_table):
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=np.array([]),
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
    )
    return model_config


def get_model_from_args(args, z_table):
    model_config = parse_args_into_model_config(args, z_table)
    model = HydroMACE(
        **model_config,
        correlation=args.correlation,
        gate=modules.gate_dict[args.gate],
        interaction_cls_first=modules.interaction_classes[args.interaction_first],
        MLP_irreps=o3.Irreps(args.MLP_irreps),
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        radial_MLP=ast.literal_eval(args.radial_MLP),
    )
    model.to(args.device)
    return model


def get_default_optimizer(
    learning_rate: float,
    model: torch.nn.Module,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
):
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        amsgrad=amsgrad,
    )
    optimizer = torch.optim.AdamW(**param_options)
    return optimizer


def setup_wandb(args):
    if args.wandb:
        logging.info("Using Weights and Biases for logging")
        import wandb

        wandb_config = {}
        args_dict = vars(args)
        args_dict_json = json.dumps(args_dict)
        for key in args.wandb_log_hypers:
            wandb_config[key] = args_dict[key]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=wandb_config,
        )
        wandb.run.summary["params"] = args_dict_json
        return True
    return False
