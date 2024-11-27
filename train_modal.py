import os
import json
import time
from datetime import datetime
from modal import Image, App, gpu, Volume, Secret
from src.bitbert.tokenizer import Tokenizer
from src.bitbert.layers import Model, ModelArgs, BitBertBlock, BertBlock
from src.bitbert.data import (
  download_dclm_data,
  download_fineweb_data,
  download_wiki_data,
  dataset_name_to_getter,
  get_combined_dataset,
  send_to_device,
)
from src.bitbert.optimizer import get_optimizer
from src.bitbert.scheduler import (
  get_wsd_scheduler,
  BatchSizeSchedule,
  MaskingScheduleCollator,
)
from src.bitbert.checkpoint import resume_from_checkpoint, save_checkpoint
from functools import partial

DATA_DIR = "/data"
MOUNT_PATH = "/training_outputs"


def download_dclm():
  download_dclm_data(DATA_DIR)


def download_fineweb():
  download_fineweb_data(DATA_DIR)


def download_wiki():
  download_wiki_data(DATA_DIR)


vol = Volume.from_name("bitbert-outputs", create_if_missing=True)

image = (
  Image.debian_slim(python_version="3.10")
  .run_commands(
    "pip install torch==2.5.0.dev20240912+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124"
  )
  .pip_install(
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
    "zstandard",
  )
  .run_function(download_dclm, secrets=[Secret.from_name("HF-SECRET")])
  .run_function(download_fineweb, secrets=[Secret.from_name("HF-SECRET")])
  .run_function(download_wiki, secrets=[Secret.from_name("HF-SECRET")])
  .pip_install("heavyball", "torchao")
  .apt_install("git")
  .pip_install("muon@git+https://github.com/KellerJordan/Muon.git")
)

app = App("train-bitbert")

block_types = {"bert": BertBlock, "bitbert": BitBertBlock}

BATCH_SIZE = 96
MAX_LR = 1.0e-3
LR_WARMUP_FRAC = 0.01
LR_DECAY_FRAC = 0.7
LR_FINAL_FRAC = 0.1
LR_FINAL_SCALE = 0.05
BATCH_SIZE_WARMUP_FRAC = 0.8
MASK_STABLE_START = 0.2
MASK_STABLE_END = 0.1
SAVE_EVERY = 20_000  # steps
DATA_SPLITS = ["dclm", "fineweb", "wiki"]
WEIGHTS = [1.0, 1.0, 0.5]
# we are now using an infinite loop, so we choose the number of samples.
# to target 300_000 steps, batch size 96, we need ~30M samples
NUM_SAMPLES = 100_000 # 60_000_000
OPTIMIZER_NAME = "muon"


@app.function(
  image=image,
  gpu=gpu.H100(),
  timeout=60 * 60 * 24,
  volumes={MOUNT_PATH: vol},
  secrets=[Secret.from_name("HF-SECRET")],
)
def train(
    job_id: str,
    block_type_name: str = "bert"
):
  import time
  import torch
  import torch.nn.functional as F
  from tqdm.auto import tqdm
  from transformers import BertForMaskedLM, BertConfig
  from liger_kernel.transformers.functional import liger_cross_entropy as cross_entropy

  print(torch.__version__)
  print("Training begins!")
  args = ModelArgs()
  block_type = block_types.get(block_type_name)
  dataset = get_combined_dataset(DATA_DIR, DATA_SPLITS, WEIGHTS)
  model = Model(block_type, args, max_seq_len=512)
  print("flex attention enabled?", model.use_flex_attention)

  # prepare checkpoint/model directory
  model_dir = f"{block_type.__name__}-{dataset}-{job_id}"
  os.makedirs(os.path.join(MOUNT_PATH, model_dir), exist_ok=True)

  device = "cuda"
  model.to(device)  # pyright: ignore
  tokenizer = Tokenizer()
  batch_size = 96
  collator = MaskingScheduleCollator(
    total_steps=NUM_SAMPLES // batch_size + 1, tokenizer=tokenizer, max_length=512
  )
  dataloader = torch.utils.data.DataLoader(
    dataset,  # pyright: ignore
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True,
    collate_fn=collator,
  )
  total_steps = NUM_SAMPLES // batch_size
  print("Total steps:", total_steps)
  optimizer = get_optimizer(model, OPTIMIZER_NAME, MAX_LR)
  scheduler = get_wsd_scheduler(
    optimizer,
    total_steps + 3,
    warmup_frac=LR_WARMUP_FRAC,
    decay_frac=LR_DECAY_FRAC,  # longer decline
    final_frac=LR_FINAL_FRAC,
    final_lr_scale=LR_FINAL_SCALE,
  )
  batch_size_manager = BatchSizeSchedule(
    BATCH_SIZE, 1536, total_steps, BATCH_SIZE_WARMUP_FRAC, BATCH_SIZE
  )
  print(
    f"Taking {batch_size_manager.steps_per_batch_size} optimizer steps per batch size"
  )

  # resume from checkpoint if it exists
  vol.reload()
  (
    model,
    optimizer,
    dataloader,
    scheduler,
    batch_size_scheduler,
    steps_so_far,
  ) = resume_from_checkpoint(
    checkpoint_dir=os.path.join(MOUNT_PATH, model_dir),
    model=model,
    optimizer=optimizer,
    dataloader=dataloader,
    scheduler=scheduler,
    batch_size_scheduler=batch_size_manager,
  )
  print("Starting from step", steps_so_far)

  losses = []
  pbar = tqdm(total=total_steps)
  pbar.update(steps_so_far)

  for i, batch in enumerate(dataloader):
    batch = send_to_device(batch, device)
    with torch.autocast(device, dtype=torch.bfloat16):
      loss = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
      )

    losses.append(loss.item())
    loss.backward()
    if batch_size_manager.step():
      optimizer.step()
      optimizer.zero_grad()
    scheduler.step()
    pbar.update(1)
    pbar.set_postfix(
      {
        "loss": round(loss.item(), 4),
        "lr": scheduler.get_last_lr()[0],
        "masking_ratio": (batch["labels"] != -100).float().sum().item()
        / batch["attention_mask"].sum().item(),
        "batch_size": batch_size_manager.get_current_batch_size(),
      }
    )
    steps_so_far += 1
    if steps_so_far >= total_steps:
      print("breaking")
      break

    if steps_so_far % SAVE_EVERY == 0:
      print("Saving checkpoint after step", steps_so_far)
      save_checkpoint(
        checkpoint_dir=os.path.join(MOUNT_PATH, model_dir),
        model=model,
        optimizer=optimizer,
        step=steps_so_far,
      )
      vol.commit()
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
