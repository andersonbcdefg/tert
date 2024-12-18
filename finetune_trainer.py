import os
import numpy as np
import torch
from src.bitbert.finetune import BERTClassifier
from src.bitbert.layers import Model, ModelArgs, BitBertBlock, BertBlock
from sklearn.metrics import accuracy_score
from modal import Image, App, gpu, Volume
from transformers import (
    Trainer,
    TrainingArguments,
)

torch.set_float32_matmul_precision("high")

MOUNT_PATH = "/training_outputs"
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
        "accelerate",
        "scipy",
        "evaluate",
    )
    .pip_install("scikit-learn")
    .copy_local_file("glue_metadata.yaml", "/glue_metadata.yaml")
)

app = App("finetune-bitbert")


@app.function(
    image=image, gpu=gpu.H100(), timeout=60 * 60 * 24, volumes={MOUNT_PATH: vol}
)
def finetune(
    checkpoint: str,  # path to the model checkpoint
    task_name: str,  # name of the task to finetune on
    max_length: int = 128,
    batch_size: int = 32,
    learning_rate: float = 1.0e-5,
):
    # get the task config

    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the RTE dataset from the GLUE benchmark
    print(f"Loading the {task_name} dataset...")

    # 2. Initialize the tokenizer for 'bert-base-uncased'
    print("Initializing the tokenizer...")

    # 5. Set the format of the datasets to PyTorch tensors
    tokenized_datasets = tokenized_datasets.rename_column(label_col, "labels")
    tokenized_datasets.set_format(  # pyright: ignore
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # 6. Split the dataset into training and validation sets
    train_dataset = tokenized_datasets["train"]  # pyright: ignore
    eval_dataset = (
        tokenized_datasets["validation"]  # pyright: ignore
        if "validation" in tokenized_datasets
        else tokenized_datasets["validation_matched"]  # pyright: ignore
    )

    print("tokenized example:", train_dataset[0])
    print([len(x) for x in train_dataset["input_ids"][:4]])  # pyright: ignore

    # 7. Initialize the BERT model for sequence classification
    print("Loading the BERT model for sequence classification...")
    # config = FineTuneConfig()
    args = ModelArgs()
    block_type = BitBertBlock if "BitBertBlock" in checkpoint else BertBlock
    model = Model(block_type, args, max_seq_len=512)
    # load checkpoint
    checkpoint_dir = os.path.join(MOUNT_PATH, checkpoint)
    if "final_model.pt" in os.listdir(checkpoint_dir):
        state_dict = torch.load(os.path.join(checkpoint_dir, "final_model.pt"))
    else:
        state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))

    model.load_state_dict(state_dict)
    classifier = BERTClassifier(model, num_classes=2, dropout=0.1)
    print("dropout set to", classifier.dropout_p)
    classifier._initialize_sep_token()
    classifier.to(device)
    classifier.train()

    # 8. Define the metric computation function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}

    # 9. Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory
        num_train_epochs=5,  # Number of training epochs
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=64,  # Batch size for evaluation
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=10,  # Log every 10 steps
        save_strategy="no",  # Disable saving checkpoints
        load_best_model_at_end=False,  # Do not load the best model at the end
        metric_for_best_model="accuracy",
        learning_rate=learning_rate,
    )

    # 10. Initialize the Trainer
    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 11. Start training
    print("Starting training...")
    trainer.train()

    # 12. Evaluate the final model on the validation set
    print("Evaluating the final model on the validation set...")
    eval_result = trainer.evaluate()
    print(f"Final Validation Accuracy: {eval_result['eval_accuracy']:.4f}")
