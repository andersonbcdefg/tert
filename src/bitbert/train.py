import os
import json
import torch
from typing import Any
from datetime import datetime
from .tokenizer import Tokenizer
from .scheduler import MaskingScheduleCollator, BatchSizeSchedule, get_wsd_scheduler
from .util import scale_grads, send_to_device
from tqdm.auto import tqdm
from pydantic import BaseModel, computed_field
from .model import Model
from .data import get_combined_dataset
from .optimizer import get_optimizer
from .checkpoint import save_checkpoint, resume_from_checkpoint
from .loss import linear_cross_entropy

# TODO: add option to resume from specific checkpoint vs.
# auto-resuming from job of the same name
class TrainArgs(BaseModel):
    microbatch_size: int = 384
    max_batch_size: int = 1536
    max_seq_len: int = 128
    max_lr: float = 3.0e-4
    lr_warmup_frac: float = 0.01
    lr_decay_frac: float = 0.7
    lr_final_frac: float = 0.01
    lr_final_scale: float = 0.05
    batch_size_warmup_frac: float = 0.9
    initial_mask_ratio: float = 0.3
    final_mask_ratio: float = 0.15
    mask_stable_start: float = 0.2
    mask_stable_end: float = 0.1
    optimizer_name: str = "muon"
    data_splits: list[str] = ["dclm", "fineweb", "wiki"]
    weights: list[float] = [1.0, 1.0, 0.5]
    num_samples: int = 100_000_000
    save_every: int = 20_000

    @computed_field
    def loose_batch_size(self) -> int:
       return int(self.microbatch_size * 1.2)

    @computed_field
    def max_flat_seq_len(self) -> int:
        '''
        this is the max possible sequence length if all the
        sequences in the batch were max_seq_len long
        '''
        return self.max_seq_len * self.microbatch_size



class TrainState(BaseModel):
    pbar: Any
    steps_so_far: int = 0
    accumulated_tokens: int = 0
    losses: list[float] = []
    batch_sizes: list[int] = []

def train_step(
    batch,
    model,
    optimizer,
    dataloader,
    scheduler,
    batch_size_manager,
    state: TrainState,
    device,
    args: TrainArgs,
):
    batch = send_to_device(batch, device)
    input_ids, labels, position_ids, block_mask = model.prepare_batch(
        batch["input_ids"], batch["attention_mask"], batch["labels"],
        args.max_flat_seq_len
    )
    with torch.autocast(device, dtype=torch.bfloat16):
        hidden_states = model(
            input_ids=input_ids,
            position_ids=position_ids,
            block_mask=block_mask,
        )
        loss = linear_cross_entropy(
            hidden_states.view(-1, hidden_states.shape[-1]),
            model.output.weight,
            labels,
            ignore_index=-100,
            reduction="sum",
        )
        num_tokens = (labels != -100).sum().item()
        state.accumulated_tokens += num_tokens
        state.batch_sizes.append(input_ids.shape[1]) # total tokens not just loss tokens
    avg_loss = loss.item() / num_tokens
    state.losses.append(avg_loss)  # mean loss on this batch
    loss.backward()
    if batch_size_manager.step():
        scale_grads(model, state.accumulated_tokens)
        optimizer.step()
        optimizer.zero_grad()
        state.accumulated_tokens = 0
    scheduler.step()
    state.pbar.update(1)
    state.pbar.set_postfix(
        {
            "loss": round(avg_loss, 4),
            "lr": scheduler.get_last_lr()[0],
            "masking_ratio": (batch["labels"] != -100).float().sum().item()
            / batch["attention_mask"].sum().item(),
            "batch_size": batch_size_manager.get_current_batch_size(),
        }
    )
    state.steps_so_far += 1

def train(
    job_id: str,
    args: TrainArgs,
    checkpoint_dir: str,  # where checkpoints are saved
    data_dir: str,  # where the data is
    save_callback=lambda: None,  # for committing the volume
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(max_seq_len=args.max_seq_len, rotary_impl="cos_sin")
    model.to(device)  # pyright: ignore
    # model = torch.compile(model)
    tokenizer = Tokenizer()
    total_steps = args.num_samples // args.microbatch_size + 1
    print("Total steps:", total_steps)
    collator = MaskingScheduleCollator(
        total_steps=total_steps, tokenizer=tokenizer, max_length=args.max_seq_len
    )
    dataset = get_combined_dataset(
        data_dir,
        args.data_splits,
        args.weights,
        chunk_size=args.max_seq_len
    )
    # we slightly increase the batch size then truncate so that every sequence
    # has a fixed shape and (probably) uses little to no padding
    dataloader = torch.utils.data.DataLoader(
        dataset,  # pyright: ignore
        batch_size=args.loose_batch_size,
        num_workers=1,
        pin_memory=True,
        collate_fn=collator,
        prefetch_factor=120
    )
    optimizer = get_optimizer(model, args.optimizer_name, args.max_lr)
    scheduler = get_wsd_scheduler(
        optimizer,
        total_steps + 3,
        warmup_frac=args.lr_warmup_frac,
        decay_frac=args.lr_decay_frac,
        final_frac=args.lr_final_frac,
        final_scale=args.lr_final_scale,
    )
    batch_size_manager = BatchSizeSchedule(
        args.microbatch_size,
        args.max_batch_size,
        total_steps,
        args.batch_size_warmup_frac,
        args.microbatch_size,
    )
    print(
        f"Taking {batch_size_manager.steps_per_batch_size} optimizer steps per batch size"
    )

    date_string = datetime.now().strftime("%m-%d-%Y")
    checkpoint_subdir = f"BertBlock-{date_string}-job-{job_id}"
    checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_subdir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    (
        model,
        optimizer,
        dataloader,
        scheduler,
        batch_size_scheduler,
        steps_so_far,
    ) = resume_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        scheduler=scheduler,
        batch_size_scheduler=batch_size_manager,
    )

    print("Starting from step", steps_so_far)
    pbar = tqdm(total=total_steps)
    pbar.update(steps_so_far)
    state = TrainState(pbar=pbar)

    for i, batch in enumerate(dataloader):
        train_step(
            batch,
            model,
            optimizer,
            dataloader,
            scheduler,
            batch_size_manager,
            state,
            device,
            args
        )

        if state.steps_so_far >= total_steps:
            print("breaking")
            break

        if state.steps_so_far % args.save_every == 0:
            print("Saving checkpoint after step", state.steps_so_far)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                step=steps_so_far,
            )
            save_callback()
    print("Training ends!")

    # save the model and metrics
    print("Saving final model")
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pt"))
    json.dump(
        {
            "model_name": checkpoint_dir,
            "losses": state.losses,
            "batch_sizes": state.batch_sizes
        },
        open(os.path.join(checkpoint_dir, "metrics.json"), "w"),
    )
    save_callback()
    print("Done!")
