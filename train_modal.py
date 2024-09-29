import os
import json
import time
from modal import Image, App, gpu, Volume
from src.bitbert.tokenizer import Tokenizer
from src.bitbert.layers import Model, ModelArgs, BitBertBlock, BertBlock
from src.bitbert.data import (
    get_wiki_dataloader,
    get_fw_dataloader,
    get_tinystories_dataloader,
    send_to_device
)
from src.bitbert.scheduler import get_wsd_scheduler, BatchSizeSchedule
from functools import partial

MOUNT_PATH = "/training_outputs"
vol = Volume.from_name("bitbert-outputs", create_if_missing=True)
dataset_to_loader = {
    "wikipedia": get_wiki_dataloader,
    "fineweb": get_fw_dataloader,
    "tinystories": get_tinystories_dataloader,
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
    "liger-kernel"
)# .run_function(get_dataloader)
# .run_commands(
#     "pip install flash-attn --no-build-isolation"
# )

app = App("train-bitbert")

@app.function(
    image=image,
    gpu=gpu.H100(),
    timeout=60 * 60 * 24,
    volumes={MOUNT_PATH: vol}
)
def train():
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
    dataset = "fineweb"
    epochs = 1
    model = Model(block_type, args, max_seq_len=512)
    print("flex attention enabled?", model.use_flex_attention)
    # config = BertConfig(vocab_size=args.n_vocab, hidden_size=args.d_model)
    # model = BertForMaskedLM(config)
    device = "cuda"
    model.to(device) # pyright: ignore
    tokenizer = Tokenizer()
    batch_size = 64
    dataloader, num_samples = dataset_to_loader[dataset](
        batch_size=batch_size, tokenizer=tokenizer
    )
    max_lr = 1.0e-4
    total_steps = epochs * num_samples // batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
    scheduler = get_wsd_scheduler(optimizer, total_steps)
    batch_size_manager = BatchSizeSchedule(128, 1536, total_steps, 0.2, batch_size)
    print(f"Taking {batch_size_manager.steps_per_batch_size} optimizer steps per batch size")
    losses = []
    pbar = tqdm(total=total_steps)
    for epoch in range(epochs):
        print(" ==== Epoch", epoch + 1, " ====")
        for i, batch in enumerate(dataloader):
            batch = send_to_device(batch, device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask']
                ) # .logits

            loss = cross_entropy(
                logits.view(-1, logits.shape[-1]),
                batch['labels'].view(-1),
                -100, 0.0, 'mean'
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
    print("Training ends!")

    # save the model and metrics

    model_dir = f"{block_type.__name__}-{dataset}-{time.time()}"
    os.makedirs(os.path.join(MOUNT_PATH, model_dir))
    torch.save(model.state_dict(), os.path.join(MOUNT_PATH, model_dir, "model.pt"))
    json.dump(
        {
            "model_name": model_dir,
            "losses": losses,
        },
        open(os.path.join(MOUNT_PATH, model_dir, "metrics.json"), "w"),
    )
