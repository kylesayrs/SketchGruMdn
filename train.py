from src.train import TrainingConfig, train_model


if __name__ == "__main__":
    config = TrainingConfig(
        wandb_mode="online",
        save_checkpoints=True,
        device="cuda",
        save_samples=True,
        latent_size=256,
    )
    train_model(config)
