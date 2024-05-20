import torch
import torch.nn.functional as F
from mace.tools import scatter

from .tools import (
    remove_selected_hydrogens_from_batch,
    sample_hydrogens_to_remove,
)


def add_noise_to_positions(batch, std: float | torch.Tensor = 0.1):
    noise = torch.randn_like(batch.positions) * std
    batch.positions += noise
    return batch


def take_step(model, batch, optimizer, args):
    batch = batch.to(args.device)
    optimizer.zero_grad()
    outputs = model(batch)
    loss = loss_function(outputs, batch, args)
    loss.backward()
    optimizer.step()
    return loss.item()


def loss_function(outputs, batch, args):
    predicted_missing_hydrogens = outputs["missing_hydrogen_logits"]
    actual_missing_hydrogens = batch["charges"].to(torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(predicted_missing_hydrogens, actual_missing_hydrogens)

    predicted_missing_per_mol = scatter.scatter_sum(
        outputs["missing_hydrogens"], batch.batch, dim=0
    ).to(torch.float32)
    actual_missing_per_mol = scatter.scatter_sum(batch.charges, batch.batch, dim=0).to(
        torch.float32
    )
    loss2 = F.mse_loss(predicted_missing_per_mol, actual_missing_per_mol)
    return loss + loss2


def calculate_validation_loss(model, val_loader, args):
    model.eval()
    validation_loss = 0
    num_atoms = 0
    with torch.no_grad():
        for batch in val_loader:
            to_remove = sample_hydrogens_to_remove(batch, full_removal_frequency=1.0)
            batch = remove_selected_hydrogens_from_batch(batch, to_remove)
            batch = batch.to(args.device)
            outputs = model(batch)
            loss = loss_function(outputs, batch, args)
            validation_loss += loss.item()
            num_atoms += batch.positions.shape[0]
    validation_loss_per_atom = validation_loss / num_atoms
    model.train()
    return validation_loss, validation_loss_per_atom
