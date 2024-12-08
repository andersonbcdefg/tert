import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score
import torch
from modal import Image, App, gpu

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
    )
    .pip_install("scikit-learn")
    .copy_local_file("glue_metadata.yaml", "/glue_metadata.yaml")
)

app = App("finetune-bitbert")


@app.function(image=image, gpu=gpu.H100(), timeout=60 * 60 * 24)
def main():
    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the RTE dataset from the GLUE benchmark
    print("Loading the RTE dataset...")
    dataset = load_dataset("glue", "rte")

    # 2. Initialize the tokenizer for 'bert-base-uncased'
    print("Initializing the tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 3. Define the tokenization function
    def tokenize_function(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    # 4. Apply the tokenization to the entire dataset
    print("Tokenizing the dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 5. Set the format of the datasets to PyTorch tensors
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # 6. Split the dataset into training and validation sets
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # 7. Initialize the BERT model for sequence classification
    print("Loading the BERT model for sequence classification...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.to(device)

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
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=64,  # Batch size for evaluation
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=10,  # Log every 10 steps
        save_strategy="no",  # Disable saving checkpoints
        load_best_model_at_end=False,  # Do not load the best model at the end
        metric_for_best_model="accuracy",
    )

    # 10. Initialize the Trainer
    trainer = Trainer(
        model=model,
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


if __name__ == "__main__":
    main()
