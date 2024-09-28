from modal import Image, App, gpu
from src.bitbert.tokenizer import Tokenizer
from src.bitbert.layers import Model, ModelArgs, BitBertBlock, BertBlock
from src.bitbert.data import get_dataloader

image = Image.debian_slim(python_version="3.10").run_commands(
    "pip install torch==2.5.0.dev20240912+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124"
).pip_install(
    "packaging", "ninja", "tiktoken", "blobfile", "transformers", "awkward", "einops", "datasets", "tqdm"
)
# .run_commands(
#     "pip install flash-attn --no-build-isolation"
# )

app = App("train-bitbert")

@app.function(image=image, gpu=gpu.H100())
def train():
    import time
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from transformers import BertForMaskedLM, BertConfig
    from torch.cuda.amp import autocast

    print(torch.__version__)
    print("Training begins!")
    args = ModelArgs()
    model = Model(BertBlock, args, max_seq_len=256)
    # config = BertConfig(vocab_size=args.n_vocab, hidden_size=args.d_model)
    # model = BertForMaskedLM(config)
    model.to("cuda") # pyright: ignore
    tokenizer = Tokenizer()
    dataloader = get_dataloader(batch_size=32, tokenizer=tokenizer)
    total_steps = len(dataloader)

    def get_lr_multiplier(step, warmup_frac=0.05):
        # warmup from 0 for the first 20% of the total steps
        warmup_steps = total_steps * warmup_frac
        cooldown_steps = total_steps * (1 - warmup_frac)
        if step < warmup_steps:
            return step / warmup_steps
        # then linearly decay from 1 to 0 for the remaining 80%
        else:
            return 1 - (step - warmup_steps) / (cooldown_steps - warmup_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_multiplier)

    accum_iters = 8
    pbar = tqdm(total=len(dataloader))
    for i, batch in enumerate(dataloader):
        # with autocast(dtype=torch.bfloat16):
        logits = model(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask']
        ) # .logits

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), batch['labels'].view(-1))
        loss.backward()
        if i % accum_iters == accum_iters - 1:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        pbar.update(1)
        pbar.set_description(f"loss: {loss.item():.4f}")
    print("Training ends!")
