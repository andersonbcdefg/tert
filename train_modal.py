import os
import json
import time
from datetime import datetime
from modal import Image, App, gpu, Volume, Secret
from src.bitbert.tokenizer import Tokenizer
from src.bitbert.layers import Model, ModelArgs, BitBertBlock, BertBlock
from src.bitbert.data import (
    get_wiki_dataloader,
    get_fw_dataloader,
    get_tinystories_dataloader,
    get_dclm_dataloader,
    send_to_device,
    get_dclm_fw_interleaved_dataloader
)
from src.bitbert.scheduler import get_wsd_scheduler, BatchSizeSchedule
from src.bitbert.checkpoint import resume_from_checkpoint, save_checkpoint
from functools import partial

DATA_DIR = "/data"
def download_dclm_data():
    import huggingface_hub
    huggingface_hub.snapshot_download(
        repo_id="TaylorAI/dclm_subset_1pct",
        repo_type="dataset",
        local_dir=DATA_DIR
    )

MOUNT_PATH = "/training_outputs"
vol = Volume.from_name("bitbert-outputs", create_if_missing=True)
dataset_to_loader = {
    "wikipedia": get_wiki_dataloader,
    "fineweb": get_fw_dataloader,
    "tinystories": get_tinystories_dataloader,
    "dclm": partial(get_dclm_dataloader, data_dir=DATA_DIR),
    "dclm_fw": partial(get_dclm_fw_interleaved_dataloader, data_dir=DATA_DIR),
}

image = Image.debian_slim(python_version="3.10").run_commands(
    "pip install torch==2.5.0.dev20240912+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124"
).pip_install(
    "packaging",
    "ninja",
    "tiktoken",
    "blobfile",
    "transformers",
    "awkward",
    "einops",
    "datasets",
    "tqdm",
    "liger-kernel",
    "pyarrow",
    "zstandard"
).run_function(download_dclm_data, secrets=[Secret.from_name("HF-SECRET")])

app = App("train-bitbert")

@app.function(
    image=image,
    gpu=gpu.H100(),
    timeout=60 * 60 * 24,
    volumes={MOUNT_PATH: vol},
    secrets=[Secret.from_name("HF-SECRET")]
)
def train(job_id: str):
    import time
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from transformers import BertForMaskedLM, BertConfig
    from liger_kernel.transformers.functional import liger_cross_entropy as cross_entropy

    print(torch.__version__)
    print("Training begins!")
    args = ModelArgs()
    block_type = BitBertBlock
    dataset = "dclm_fw"
    epochs = 1
    model = Model(block_type, args, max_seq_len=512)
    print("flex attention enabled?", model.use_flex_attention)

    # prepare checkpoint/model directory
    model_dir = f"{block_type.__name__}-{dataset}-{job_id}"
    os.makedirs(os.path.join(MOUNT_PATH, model_dir), exist_ok=True)

    device = "cuda"
    model.to(device) # pyright: ignore
    tokenizer = Tokenizer()
    batch_size = 96
    dataloader, num_samples = dataset_to_loader[dataset](
        batch_size=batch_size, tokenizer=tokenizer
    )
    max_lr = 1.0e-4
    total_steps = epochs * num_samples // batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
    scheduler = get_wsd_scheduler(optimizer, total_steps + 3, warmup_frac=0.02) # just in case
    batch_size_manager = BatchSizeSchedule(96, 1536, total_steps, 0.9, batch_size)
    print(f"Taking {batch_size_manager.steps_per_batch_size} optimizer steps per batch size")

    # resume from checkpoint if it exists
    vol.reload()
    (
        model,
        optimizer,
        dataloader,
        scheduler,
        batch_size_scheduler,
        steps_so_far
    ) = resume_from_checkpoint(
        checkpoint_dir=os.path.join(MOUNT_PATH, model_dir),
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        scheduler=scheduler,
        batch_size_scheduler=batch_size_manager
    )
    print("Starting from step", steps_so_far)

    SAVE_EVERY = 5_000 # steps
    losses = []
    pbar = tqdm(total=total_steps - steps_so_far)
    for epoch in range(epochs):
        print(" ==== Epoch", epoch + 1, " ====")
        for i, batch in enumerate(dataloader):
            batch = send_to_device(batch, device)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            losses.append(loss.item())
            loss.backward()
            if batch_size_manager.step():
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            pbar.update(1)
            pbar.set_postfix({
                "loss": round(loss.item(), 4),
                "lr": scheduler.get_last_lr()[0],
                "batch_size": batch_size_manager.get_current_batch_size(),
            })
            steps_so_far += 1
            if steps_so_far >= total_steps:
                break

            if steps_so_far % SAVE_EVERY == 0:
                print("Saving checkpoint after step", steps_so_far)
                save_checkpoint(
                    checkpoint_dir=os.path.join(MOUNT_PATH, model_dir),
                    model=model,
                    optimizer=optimizer,
                    step=steps_so_far
                )
                vol.commit()
        if steps_so_far >= total_steps:
            break
    print("Training ends!")

    # save the model and metrics
    print("Saving final model")
    torch.save(model.state_dict(), os.path.join(MOUNT_PATH, model_dir, "final_model.pt"))
    json.dump(
        {
            "model_name": model_dir,
            "losses": losses,
        },
        open(os.path.join(MOUNT_PATH, model_dir, "metrics.json"), "w"),
    )
    vol.commit()
    print("Done!")
