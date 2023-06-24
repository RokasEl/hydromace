import torch


def add_noise_to_positions(batch, std: float | torch.Tensor = 0.1):
    noise = torch.randn_like(batch.positions)
    noise = noise/ torch.norm(noise, dim=-1, keepdim=True)*std
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
    predicted_missing_hydrogens = outputs["missing_hydrogens"]
    actual_missing_hydrogens = batch["charges"]
    loss = (predicted_missing_hydrogens - actual_missing_hydrogens).pow(2).mean()
    return loss


def calculate_validation_loss(model, val_loader, args):
    model.eval()
    validation_loss = 0
    num_atoms = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(args.device)
            outputs = model(batch)
            loss = loss_function(outputs, batch, args)
            validation_loss += loss.item()
            num_atoms += batch.positions.shape[0]
    validation_loss_per_atom = validation_loss / num_atoms
    model.train()
    return validation_loss, validation_loss_per_atom
