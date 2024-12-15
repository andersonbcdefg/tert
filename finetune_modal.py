import yaml

import os
import json
from dataclasses import dataclass
from modal import Image, App, gpu, Volume
import torch
from src.bitbert.scheduler import get_wsd_scheduler
from src.bitbert.finetune import BERTClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm
from src.bitbert.finetune import finetune, finetune_hf, FineTuneArgs

MOUNT_PATH = "/training_outputs"
vol = Volume.from_name("bitbert-outputs", create_if_missing=True)

# image = (
#     Image.debian_slim(python_version="3.10")
#     .run_commands(
#         "pip install torch==2.5.0.dev20240912+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124"
#     )
#     .pip_install(
#         "packaging",
#         "ninja",
#         "tiktoken",
#         "blobfile",
#         "transformers",
#         "awkward",
#         "einops",
#         "datasets",
#         "tqdm",
#         "liger-kernel",
#     )
#     .pip_install("scikit-learn")
#     .copy_local_file("glue_metadata.yaml", "/glue_metadata.yaml")
# )
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "pip install torch==2.6.0.dev20241119+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124"
    )
    .pip_install(
        "datasets==3.1.0",
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
        "pydantic>=2.0",
        "adam-mini"
    )
    .pip_install(
        "torchao@git+https://github.com/pytorch/ao.git",
        "muon@git+https://github.com/KellerJordan/Muon.git",
        "cut-cross-entropy@git+https://github.com/andersonbcdefg/ml-cross-entropy.git"
    ).pip_install("scikit-learn")
    # .run_function(download_wiki, secrets=[Secret.from_name("HF-SECRET")])
    # .run_function(download_dclm, secrets=[Secret.from_name("HF-SECRET")])
    # .run_function(download_fineweb, secrets=[Secret.from_name("HF-SECRET")])
    # .env({"TORCH_TRACE": "/training_outputs/trace"})
)

app = App("finetune-bitbert")

@app.function(
    image=image,
    gpu=gpu.H100(),
    timeout=60 * 60 * 24,
    volumes={MOUNT_PATH: vol}
)
def run_finetune(
    checkpoint: str,  # path to the model checkpoint
    task: str = "RTE"
):
    import torch
    from tqdm.auto import tqdm
    print(torch.__version__)
    for lr in [1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 5.0e-4, 1.0e-3]:
        print("=== RUNNING LR", lr, "===")
        config = FineTuneArgs(
            learning_rate=lr,
            checkpoint=checkpoint# os.path.join(MOUNT_PATH, checkpoint)
        )
        result = finetune_hf(config)
        os.makedirs(os.path.join(MOUNT_PATH, f"12142024_finetune_hf"), exist_ok=True)
        json.dump(
            result,
            open(os.path.join(MOUNT_PATH, f"12142024_finetune_hf/lr_{lr}.json"), "w")
        )
    vol.commit()
    # args = ModelArgs()
    # block_type = BitBertBlock if "BitBertBlock" in checkpoint else BertBlock
    # model = Model(block_type, args, max_seq_len=512)
    # # load checkpoint
    # glue_metadata = yaml.safe_load(open("/glue_metadata.yaml"))
    # checkpoint_dir = os.path.join(MOUNT_PATH, checkpoint)
    # if "final_model.pt" in os.listdir(checkpoint_dir):
    #     state_dict = torch.load(os.path.join(checkpoint_dir, "final_model.pt"))
    # else:
    #     state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))

    # model.load_state_dict(state_dict)

    # # After loading the state dict
    # model_keys = set(model.state_dict().keys())
    # loaded_keys = set(state_dict.keys())
    # missing_keys = model_keys - loaded_keys
    # unexpected_keys = loaded_keys - model_keys
    # print("missing:", missing_keys)
    # print("unexpected:", unexpected_keys)

    # if missing_keys:
    #     print(f"Warning: Missing keys in loaded state dict: {missing_keys}")
    # if unexpected_keys:
    #     print(f"Warning: Unexpected keys in loaded state dict: {unexpected_keys}")

    # classifier = BERTClassifier(model, num_classes=glue_metadata["num_classes"][task])
    # print("flex attention enabled?", model.use_flex_attention)
    # # config = BertConfig(vocab_size=args.n_vocab, hidden_size=args.d_model)
    # # model = BertForMaskedLM(config)
    # device = "cuda"
    # classifier.to(device)  # pyright: ignore
    # tokenizer = Tokenizer()
    # classifier._initialize_sep_token(token_ids_to_average=[tokenizer.eos_id])

    # print(
    #     "trainable params:",
    #     sum(p.numel() for p in classifier.parameters() if p.requires_grad),
    # )
    # print(
    #     "trainable params list:",
    #     [n for n, p in classifier.named_parameters() if p.requires_grad],
    # )

    # # print(glue_metadata)
    # train_dataloader, dev_dataloaders = get_glue_dataloaders(
    #     glue_metadata, task, batch_size=config.batch_size, tokenizer=tokenizer
    # )
    # print(len(train_dataloader))
    # total_steps = config.num_epochs * len(train_dataloader)
    # optimizer = torch.optim.AdamW(
    #     classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay
    # )
    # scheduler = get_wsd_scheduler(
    #     optimizer, total_steps, warmup_frac=0.1, decay_frac=0.9
    # )
    # losses = []
    # val_results = []
    # print("Finetuning begins!")
    # pbar = tqdm(total=total_steps)
    # for epoch in range(config.num_epochs):
    #     print(" ==== Epoch", epoch + 1, " ====")
    #     print("Evaluating model on dev set...")
    #     result = eval_model(
    #         classifier,
    #         dev_dataloaders[0],
    #         glue_metadata["num_classes"][task],
    #         metrics=glue_metadata["metrics"][task],
    #         device=device,
    #     )
    #     print(f"Dev {task} results after {epoch} epochs:")
    #     print(result)
    #     val_results.append(result)
    #     for i, batch in enumerate(train_dataloader):
    #         batch = send_to_device(batch, device)
    #         with torch.autocast(device, dtype=torch.bfloat16):
    #             loss = classifier(
    #                 input_ids=batch["input_ids"],
    #                 attention_mask=batch["attention_mask"],
    #                 labels=batch["labels"],
    #                 dropout_p=config.dropout,
    #             )  # .logits

    #         losses.append(loss.item())
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         scheduler.step()
    #         pbar.update(1)
    #         pbar.set_postfix(
    #             {"loss": round(loss.item(), 4), "lr": scheduler.get_last_lr()[0]}
    #         )
    # result = eval_model(
    #     classifier,
    #     dev_dataloaders[0],
    #     glue_metadata["num_classes"][task],
    #     metrics=glue_metadata["metrics"][task],
    #     device=device,
    # )
    # print(f"Dev {task} results after {config.num_epochs} epochs:")
    # print(result)
    # val_results.append(result)
    # print("Training ends!")

    # print(val_results)

    # return losses, val_results


# @app.function(
#     image=image,
#     gpu=gpu.H100(),
#     timeout=60 * 60 * 24,
#     volumes={MOUNT_PATH: vol}
# )
# def finetune_hf(
#     task: str,
#     model_path: str = "bert-base-uncased", # path to the model checkpoint
# ):
#     import time
#     import torch
#     import torch.nn.functional as F
#     from tqdm.auto import tqdm
#     from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer
#     glue_metadata = yaml.safe_load(open("/glue_metadata.yaml"))
#     config = FineTuneConfig()
#     classifier = BertForSequenceClassification.from_pretrained(
#         model_path,
#         num_labels=glue_metadata['num_classes'][task],
#         hidden_dropout_prob=config.dropout,
#         attention_probs_dropout_prob=config.dropout
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     print(torch.__version__)
#     print("Finetuning begins!")

#     device = "cuda"
#     classifier.to(device) # pyright: ignore
#     train_dataloader, dev_dataloaders = get_glue_dataloaders(
#         glue_metadata,
#         task,
#         batch_size=config.batch_size,
#         tokenizer=tokenizer
#     )
#     print(len(train_dataloader))
#     total_steps = config.num_epochs * len(train_dataloader)
#     optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)
#     scheduler = get_wsd_scheduler(optimizer, total_steps, decay_frac=0.0)
#     losses = []
#     val_results = []
#     pbar = tqdm(total=total_steps)
#     for epoch in range(config.num_epochs):
#         print(" ==== Epoch", epoch + 1, " ====")
#         print("Evaluating model on dev set...")
#         result = eval_model(
#             classifier,
#             dev_dataloaders[0],
#             glue_metadata['num_classes'][task],
#             metrics=glue_metadata['metrics'][task],
#             device=device
#         )
#         print(f"Dev {task} results after {epoch} epochs:")
#         print(result)
#         val_results.append(result)
#         for i, batch in enumerate(train_dataloader):
#             batch = send_to_device(batch, device)

#             loss = classifier(
#                 input_ids=batch['input_ids'],
#                 attention_mask=batch['attention_mask'],
#                 labels=batch['labels']
#             ) # .logits
#             if hasattr(loss, "loss"):
#                 loss = loss.loss

#             losses.append(loss.item())
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             scheduler.step()
#             pbar.update(1)
#             pbar.set_postfix({
#                 "loss": round(loss.item(), 4),
#                 "lr": scheduler.get_last_lr()[0]
#             })
#     result = eval_model(
#         classifier,
#         dev_dataloaders[0],
#         glue_metadata['num_classes'][task],
#         metrics=glue_metadata['metrics'][task],
#         device=device
#     )
#     print(f"Dev {task} results after {config.num_epochs} epochs:")
#     print(result)
#     val_results.append(result)
#     print("Training ends!")

#     print(val_results)

#     return losses, val_results
