import argparse
import pyplugs
import json
import os

import torch

# huggingface libraries
import transformers
import tasks.imdb.models as models
import datasets 

model_names = ["bert-base-cased"] # todo

normalize = "todo"

train_transforms = "todo"

validation_transforms = "todo"

raw_datasets = datasets.load_dataset("imdb") # todo: generalize this
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased") # todo: generalize this

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

preprocess = "todo" # todo validation_transforms

@pyplugs.register
def train_dataset(data_dir): # todo: use data_dir
    return tokenized_datasets["train"]

@pyplugs.register
def validation_dataset(data_dir): # todo: use data_dir
    return tokenized_datasets["test"]

@pyplugs.register
def default_epochs():
    return 3

@pyplugs.register
def default_initial_lr():
    return 5e-5

@pyplugs.register
def default_lr_scheduler(optimizer, num_epochs, steps_per_epoch, start_epoch=0):
    # todo: update with start_epoch
    num_training_steps = num_epochs * steps_per_epoch 
    return transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

@pyplugs.register
def default_optimizer(model, lr, momentum, weight_decay):
    return transformers.AdamW(model.parameters(), lr,
                              weight_decay=weight_decay)

@pyplugs.register
def to_device(batch, device, gpu_id):
    # todo: deal with gpu_id
    return {k: v.to(device) for k, v in batch.items()}

@pyplugs.register
def get_input(batch):
    return batch

@pyplugs.register
def get_loss(output, batch, criterion):
    del criterion # not using it
    return output.loss

metrics = "todo"

idx2label = "todo"