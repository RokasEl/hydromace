import logging
from pathlib import Path

import ase.io as aio
import mace
import torch
from mace import modules
from mace.tools import (
    MetricsLogger,
    build_default_arg_parser,
    get_atomic_number_table_from_zs,
    get_tag,
    set_default_dtype,
    set_seeds,
    setup_logger,
)

import wandb
from hydromace.tools import (
    remove_selected_hydrogens_from_batch,
    sample_hydrogens_to_remove,
)
from hydromace.training import (
    add_noise_to_positions,
    calculate_validation_loss,
    take_step,
)
from hydromace.training_tools import (
    get_dataloaders,
    get_default_optimizer,
    get_model_from_args,
    handle_e3nn_args,
    setup_wandb,
)


def main():
    parser = build_default_arg_parser()
    parser.add_argument(
        "--noise_scale", help="Scale of rattling", default=0.4, type=float
    )
    args = parser.parse_args()
    tag = get_tag(name=args.name, seed=args.seed)
    # Setup
    set_seeds(args.seed)
    setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    logging.info(f"Running with args: {args}")
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    set_default_dtype(args.default_dtype)
    # Load data
    train_valid_data = aio.read(args.train_file, index=":", format="extxyz")
    logging.info(f"Train file loaded with {len(train_valid_data)} entries")
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
    total_steps = len(train_loader) * args.max_num_epochs
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps
    )
    use_wandb = setup_wandb(args)
    metrics_logger = MetricsLogger(directory=args.results_dir, tag=tag + "_train")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)
    for epoch in range(args.max_num_epochs):
        for batch in train_loader:
            with torch.no_grad():
                to_remove = sample_hydrogens_to_remove(batch)
                batch = remove_selected_hydrogens_from_batch(batch, to_remove)
            noise_level = (
                torch.rand((batch.positions.shape[0], 1), generator=rng)
                * args.noise_scale
            )
            batch = add_noise_to_positions(batch, noise_level)
            loss = take_step(model, batch, optimizer, args)
            if use_wandb:
                wandb.log({"loss": loss})
            lr_scheduler.step()
        if epoch % args.eval_interval == 0:
            total_loss, per_atom_loss = calculate_validation_loss(
                model, val_loader, args
            )
            dump = {
                "epoch": epoch,
                "lr": lr_scheduler.get_last_lr()[0],
                "total_loss": total_loss,
                "per_atom_loss": per_atom_loss,
            }
            metrics_logger.log(dump)
            if use_wandb:
                wandb.log(dump)
            logging.info(f"Epoch {epoch} validation loss: {per_atom_loss}")

    # Evaluation on test datasets
    logging.info("Computing metrics for training, validation, and test sets")
    if args.test_file is not None:
        test_data = aio.read(args.test_file, index=":", format="extxyz")
        test_loader, _ = get_dataloaders(
            test_data,
            args.batch_size,
            z_table,
            args.charges_key,
            args.r_max,
            -1,
        )
        _, test_per_atom_loss = calculate_validation_loss(model, test_loader, args)
    else:
        logging.info("No test file provided, skipping test set evaluation")
        test_per_atom_loss = None

    _, train_per_atom_loss = calculate_validation_loss(model, train_loader, args)
    _, val_per_atom_loss = calculate_validation_loss(model, val_loader, args)
    logging.info(
        f"Training set loss: {train_per_atom_loss}, validation set loss: {val_per_atom_loss}, test set loss: {test_per_atom_loss}"
    )

    torch.save(model, Path(args.model_dir) / (args.name + ".model"))
    logging.info("Done")


if __name__ == "__main__":
    main()
