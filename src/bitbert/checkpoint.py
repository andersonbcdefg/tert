import os
from tqdm.auto import trange
import torch
import torch.nn as nn
from typing import Any


def resume_from_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Any,
    dataloader: Any,
    scheduler: Any,
    batch_size_scheduler: Any,
):
    if not os.path.exists(os.path.join(checkpoint_dir, "checkpoint.pt")):
        print("No checkpoint found; starting from the beginning.")
        return (model, optimizer, dataloader, scheduler, batch_size_scheduler, 0)
    else:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # fastforward the dataloader
        print(
            f"Resuming from checkpoint, fast-forwarding by {checkpoint['step']} steps..."
        )
        dataloader_iter = iter(dataloader)
        for i in trange(
            checkpoint["step"], desc="Fast-forwarding dataloader & scheduler..."
        ):
            next(dataloader_iter)
            scheduler.step()
            batch_size_scheduler.step()
        return (
            model,
            optimizer,
            dataloader,
            scheduler,
            batch_size_scheduler,
            checkpoint["step"],
        )


def save_checkpoint(checkpoint_dir: str, model: nn.Module, optimizer: Any, step: int):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    # overwrites previous checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint.pt"))
