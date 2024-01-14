import os
import torch
import wandb
import pickle

from src.train.config import TrainingConfig
from src.model import AutoEncoder
from src.data import QuickdrawStrokeDataset, load_drawings_strokes, split_drawings_strokes
from src.train import create_optimizer, batch_callback


def train_model(config: TrainingConfig):
    run = wandb.init(
        project="SketchCNN-AE",
        entity="kylesayrs",
        name=None,
        reinit=True,
        mode=config.wandb_mode,
        config=config.dict()
    )
    print(f"Run id: {wandb.run.id}")
    print(config)

    print("loading data")
    # load data
    if os.path.exists("drawings_strokes.pkl") and os.path.exists("index_lookup.pkl"):
        with open("drawings_strokes.pkl", "rb") as file:
            drawings_strokes = pickle.load(file)

        with open("index_lookup.pkl", "rb") as file:
            index_lookup = pickle.load(file)
    else:
        drawings_strokes, index_lookup = load_drawings_strokes(config.data_dir)

        with open("drawings_strokes.pkl", "wb") as file:
            pickle.dump(drawings_strokes, file)

        with open("index_lookup.pkl", "wb") as file:
            pickle.dump(index_lookup, file)

    # split data
    train_index_lookup, test_index_lookup = split_drawings_strokes(
        index_lookup, test_size=0.2, shuffle=True
    )

    # data loaders
    train_loader = torch.utils.data.DataLoader(
        QuickdrawStrokeDataset(drawings_strokes, train_index_lookup, image_size=config.image_size),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        pin_memory_device=config.device
    )
    test_loader = torch.utils.data.DataLoader(
        QuickdrawStrokeDataset(drawings_strokes, test_index_lookup, image_size=config.image_size),
        batch_size=config.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        pin_memory_device=config.device
    )

    # create model, optimizer, and loss
    model = AutoEncoder(config.image_size, config.latent_size)
    optimizer = create_optimizer(model, config.optimizer, lr=config.lr)
    criterion = torch.nn.MSELoss()

    # data parallel and device loading
    if config.device_ids is not None:
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    model = model.to(config.device)
    criterion = criterion.to(config.device)

    # train model
    for epoch_index in range(config.num_epochs):
        for batch_index, images in enumerate(train_loader):
            # load data to device
            images = images.to(config.device)

            # forward
            optimizer.zero_grad()
            reconstructions, _latents = model(images)

            # backwards
            loss = criterion(images, reconstructions)
            loss.backward()

            # optimize
            optimizer.step()

            # log
            batch_callback(
                config,
                model,
                test_loader,
                len(train_loader),
                criterion,
                loss.item(),
                epoch_index,
                batch_index
            )
