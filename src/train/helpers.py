import os
import wandb
import torch
import matplotlib.pyplot as plt

from src.train import TrainingConfig


def create_optimizer(
    model: torch.nn.Module,
    optimizer: str,
    **optimizer_kwargs
) -> torch.optim.Optimizer:
    match optimizer.lower():
        case "adam":
            return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        
    raise ValueError(f"Unknown optimizer {optimizer}")


def batch_callback(
    config: TrainingConfig,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_batches: int,
    criterion: torch.nn.Module,
    train_loss: float,
    epoch_index: int,
    batch_index: int
):
    total_num_batches = epoch_index * num_batches + batch_index
    if total_num_batches % config.logging_rate == 0:
        test_loss = test_model(config, model, test_loader, criterion)
        log_metrics(config, epoch_index, batch_index, num_batches, train_loss, test_loss)

        if config.save_samples:
            save_sample(config, epoch_index, batch_index, model, test_loader)

    if config.save_checkpoints and total_num_batches % config.save_checkpoints_rate == 0:
        save_checkpoint(model, epoch_index, batch_index)


def save_checkpoint(
    model: torch.nn.Module,
    epoch_index: int,
    batch_index: int,
):
    dir_path = os.path.join("checkpoints", f"{wandb.run.id}")
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"checkpoint_{epoch_index}_{batch_index}.pth"
    save_path = os.path.join(dir_path, file_name)

    torch.save(model.state_dict(), save_path)


def save_sample(
    config: TrainingConfig,
    epoch_index: int,
    batch_index: int,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
):
    image = next(iter(test_loader))[0: 1]
    image = image.to(config.device)

    with torch.no_grad():
        reconstruction, _latent = model(image)

    image = image.cpu().numpy().squeeze(0).squeeze(0)
    reconstruction = reconstruction.cpu().numpy().squeeze(0).squeeze(0)

    figure, axes = plt.subplots(1, 2)
    axes[0].imshow(image, vmin=0.0, vmax=1.0)
    axes[1].imshow(reconstruction, vmin=0.0, vmax=1.0)

    dir_path = os.path.join("samples", f"{wandb.run.id}")
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"sample_{epoch_index}_{batch_index}.png"
    save_path = os.path.join(dir_path, file_name)
    figure.savefig(save_path)
    plt.close(figure)


def log_metrics(
    config: TrainingConfig,
    epoch_index: int,
    batch_index: int,
    num_batches: int,
    train_loss: float,
    test_loss: float
):  
    train_loss_normed = train_loss / config.batch_size
    test_loss_normed = test_loss / config.batch_size

    wandb.log({
        "train_loss": train_loss_normed,
        "test_loss": test_loss_normed,
    })

    print(
        f"[{epoch_index} / {config.num_epochs}] "
        f"[{batch_index} / {num_batches}] "
        f"train_loss: {train_loss_normed:.7f} "
        f"test_loss: {test_loss_normed:.7f} "
    )


def test_model(
    config: TrainingConfig,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module
) -> float:
    test_images = next(iter(test_loader))
    test_images = test_images.to(config.device)

    with torch.no_grad():
        model.eval()
        test_reconstructions, _latents = model(test_images)
        test_loss = criterion(test_images, test_reconstructions)
        model.train()

    return test_loss.item()
