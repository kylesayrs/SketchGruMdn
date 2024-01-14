from typing import List, Optional
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    data_dir: str = Field(default="drawings")
    image_size: int = Field(default=50)
    test_size: float = Field(default=0.2)

    latent_size: int = Field(default=256)

    num_epochs: int = Field(default=2)
    batch_size: int = Field(default=256)
    lr: float = Field(default=1e-4)
    optimizer: str = Field(default="adam")
    momentum: float = Field(default=0.95)

    logging_rate: int = Field(default=100)
    test_batch_size: int = Field(default=128)
    wandb_mode: str = Field(default="online")
    save_samples: bool = Field(default=False)

    save_checkpoints: bool = Field(default=True)
    save_checkpoints_rate: int = Field(default=1_500)

    device: str = Field(default="cpu")
    device_ids: Optional[List[int]] = Field(default=None)
