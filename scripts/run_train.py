import logging
from pathlib import Path

import ase.io as aio
import mace
from mace import modules
from mace.tools import (
    MetricsLogger,
    build_default_arg_parser,
    get_atomic_number_table_from_zs,
    get_tag,
    init_device,
    set_default_dtype,
    set_seeds,
    setup_logger,
)
from mace.tools.scripts_utils import LRScheduler

import wandb
from hydromace.training import (
    add_noise_to_positions,
    calculate_validation_loss,
    take_step,
)
from hydromace.training_tools import *


def main():
    parser = build_default_arg_parser()
    args = parser.parse_args()
    tag = get_tag(name=args.name, seed=args.seed)
    # Setup
    set_seeds(args.seed)
    setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    args.device = init_device(args.device)
    set_default_dtype(args.default_dtype)
    # Load data
    train_valid_data = aio.read(args.train_file, index=":", format="extxyz")
    z_table = get_atomic_number_table_from_zs(
        z for atoms in train_valid_data for z in atoms.get_atomic_numbers()
    )

    train_loader, val_loader = get_dataloaders(
        train_valid_data,
        args.batch_size,
        z_table,
        args.charges_key,
        args.r_max,
        args.valid_fraction,
        args.seed,
    )

    if args.compute_avg_num_neighbors:
        args.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)

    args = handle_e3nn_args(args)
    logging.info(f"Hidden irreps: {args.hidden_irreps}")

    model = get_model_from_args(args, z_table)

    optimizer = get_default_optimizer(
        learning_rate=args.lr,
        model=model,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )
    lr_scheduler = LRScheduler(optimizer, args)
    use_wandb = setup_wandb(args)
    metrics_logger = MetricsLogger(directory=args.results_dir, tag=tag + "_train")
    rng = torch.Generator(device=args.device)
    rng.manual_seed(args.seed)
    for epoch in range(args.max_num_epochs):
        for batch in train_loader:
            noise_level = (
                torch.rand(batch.positions.shape[0], generator=rng, device=args.device)
                * 0.5
            )
            batch = add_noise_to_positions(batch, noise_level)
            take_step(model, batch, optimizer, lr_scheduler, args)
        if epoch % args.eval_interval == 0:
            total_loss, per_atom_loss = calculate_validation_loss(
                model, val_loader, args
            )
            dump = {
                "epoch": epoch,
                "total_loss": total_loss,
                "per_atom_loss": per_atom_loss,
            }
            metrics_logger.log(dump)
            if use_wandb:
                wandb.log(dump)
            logging.info(f"Epoch {epoch} validation loss: {per_atom_loss}")

    # Evaluation on test datasets
    logging.info("Computing metrics for training, validation, and test sets")
    test_data = aio.read(args.test_file, index=":", format="extxyz")
    test_loader, _ = get_dataloaders(
        test_data,
        args.batch_size,
        z_table,
        args.charges_key,
        args.r_max,
        -1,
    )
    _, train_per_atom_loss = calculate_validation_loss(model, train_loader, args)
    _, val_per_atom_loss = calculate_validation_loss(model, val_loader, args)
    _, test_per_atom_loss = calculate_validation_loss(model, test_loader, args)
    logging.info(
        f"Training set loss: {train_per_atom_loss}, validation set loss: {val_per_atom_loss}, test set loss: {test_per_atom_loss}"
    )

    torch.save(model, Path(args.model_dir) / (args.name + ".model"))
    logging.info("Done")


if __name__ == "__main__":
    main()
