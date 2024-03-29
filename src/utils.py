# -*- coding: utf-8 -*-
"""indic-top experiments huggingface accelerate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jrbpsdDRAhvor7JOnTZnao3ylCtt2eMJ

# Install and Import Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2


import random
import sys
import requests
import warnings
import csv
import itertools
import math
import numpy as np
from accelerate.logging import get_logger
from accelerate import Accelerator
from transformers import (
    AutoModelForSeq2SeqLM,
    EncoderDecoderModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
    default_data_collator,
)
from torch.utils.data import DataLoader
from torch.nn import functional as F
import sentencepiece as spm
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import evaluate as evaluator
import pandas as pd
import transformers
import Levenshtein as lev
from tqdm import tqdm
import argparse
import glob
import re
import string
from collections import Counter
import os
import datasets
import logging
import json
import torch
from intents_slots import intents_slots
from evaluations import *
from configs import hyperparameters, model_step_size

print(torch.__version__)


torch.backends.cuda.matmul.allow_tf32 = True

if torch.cuda.is_available():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# import tensorflow as tf


# from tqdm.notebook import tqdm


logger = get_logger(__name__)

"""# Prepare Dataset"""

# Load a metric
metric = evaluator.load("exact_match")

# Summary info about the Metric
# metric

"""# Define the Translation Data Module"""

mbart_dict = {
    "hi": "hi_IN",
    "bn": "bn_IN",
    "mr": "mr_IN",
    "as": "bn_IN",
    "ta": "ta_IN",
    "te": "te_IN",
    "or": "bn_IN",
    "ml": "ml_IN",
    "pa": "hi_IN",
    "gu": "gu_IN",
    "kn": "te_IN",
    "en": "en_XX",
}

INDIC = ["hi", "bn", "mr", "as", "ta", "te", "or", "ml", "pa", "gu", "kn"]




def create_dataset(dataset, train_lang, test_lang, lang=None, backtranslated=False, orig=False):

    if train_lang != "en":
        with open(f"Indic-SemParse/filtered_data/{dataset}/{train_lang}.json", "r") as f:
            train_data = json.load(f)

    else:
        if lang is not None:
            with open(f"Indic-SemParse/filtered_data/{dataset}/{lang}.json", "r") as f:
                train_data = json.load(f)
        else:
            with open(f"Indic-SemParse/filtered_data/{dataset}/hi.json", "r") as f:
                train_data = json.load(f)

    if test_lang != "en":
        with open(f"Indic-SemParse/filtered_data/{dataset}/{test_lang}.json", "r") as f:
            test_data = json.load(f)

    else:
        if lang is not None:
            with open(f"Indic-SemParse/filtered_data/{dataset}/{lang}.json", "r") as f:
                test_data = json.load(f)
        else:
            with open(f"Indic-SemParse/filtered_data/{dataset}/hi.json", "r") as f:
                test_data = json.load(f)

    train = train_data["train"]

    val = test_data["val"]

    test = test_data["test"]
    

    if dataset == "itop":
        for idx, example in enumerate(train):
            if train_lang.startswith("hi_"):
                train[idx]["src"] = example["translated text"]
                
            elif train_lang != "en":
                train[idx]["src"] = example["postprocessed_translated_question"]
                
            else:
                if lang is not None:
                    train[idx]["src"] = example["text"]
                else:
                    train[idx]["src"] = example["question"]
                    
            trg = "decoupled logical form" if "decoupled logical form" in example else "logical_form"
            train[idx]["trg"] = example[trg]

        for idx, example in enumerate(val):
            if test_lang.startswith("hi_"):
                if backtranslated and not orig:
                    val[idx]["src"] = example["back translated text"]
                else:
                    val[idx]["src"] = example["translated text"]
                    
            elif test_lang != "en":
                if backtranslated and not orig:
                    val[idx]["src"] = example["backtranslated_post_processed_questions"]
                else:
                    val[idx]["src"] = example["postprocessed_translated_question"]
                
            else:
                if lang is not None:
                    val[idx]["src"] = example["text"]
                else:
                    val[idx]["src"] = example["question"]
            trg = "decoupled logical form" if "decoupled logical form" in example else "logical_form"
            val[idx]["trg"] = example[trg]

        for idx, example in enumerate(test):
            if test_lang.startswith("hi_"):
                if backtranslated and not orig:
                    test[idx]["src"] = example["back translated text"]
                else:
                    test[idx]["src"] = example["translated text"]
                    
            elif test_lang != "en":
                if backtranslated:
                    test[idx]["src"] = example[
                        "backtranslated_post_processed_questions"
                    ]
                else:
                    test[idx]["src"] = example["postprocessed_translated_question"]
                    
            else:
                if lang is not None:
                    test[idx]["src"] = example["text"]
                else:
                    test[idx]["src"] = example["question"]
            trg = "decoupled logical form" if "decoupled logical form" in example else "logical_form"
            test[idx]["trg"] = example[trg]

    elif dataset == "indic-TOP":
        for idx, example in enumerate(train):
            if train_lang != "en" and not orig:
                train[idx]["src"] = example["postprocessed_translated_question"]
            elif orig:
                train[idx]["src"] = example["original question"]
            else:
                train[idx]["src"] = example["question"]
            train[idx]["trg"] = example["decoupled logical form"]

        for idx, example in enumerate(val):
            if test_lang != "en" or orig:
                if backtranslated:
                    val[idx]["src"] = example["backtranslated_post_processed_questions"]
                elif orig:
                    val[idx]["src"] = example["original question"]
                else:
                    val[idx]["src"] = example["postprocessed_translated_question"]
            else:
                val[idx]["src"] = example["question"]
            val[idx]["trg"] = example["decoupled logical form"]

        for idx, example in enumerate(test):
            if test_lang != "en" or orig:
                if backtranslated:
                    test[idx]["src"] = example[
                        "backtranslated_post_processed_questions"
                    ]
                elif orig:
                    test[idx]["src"] = example["original question"]
                else:
                    test[idx]["src"] = example["postprocessed_translated_question"]
            else:
                test[idx]["src"] = example["question"]
            test[idx]["trg"] = example["decoupled logical form"]
            

    elif dataset == "indic-atis":
        for idx, example in enumerate(train):
            if train_lang.startswith("hi_") or (lang is not None and lang.startswith("hi_")):
                train[idx]["src"] = example["translated text"]
            
            elif train_lang != "en":
                train[idx]["src"] = example["translated text"]
            
            else:
                train[idx]["src"] = example["text"]
            trg = "decoupled logical form" if "decoupled logical form" in example else "logical form"
            train[idx]["trg"] = example[trg]

        for idx, example in enumerate(val):
            if test_lang.startswith("hi_") or (lang is not None and lang.startswith("hi_")):
                if backtranslated:
                    val[idx]["src"] = example["back translated text"]
                else:
                    val[idx]["src"] = example["translated text"]
                        
            elif test_lang != "en":
                if backtranslated:
                    val[idx]["src"] = example["back translated text"]
                else:
                    val[idx]["src"] = example["translated text"]
                    
            else:
                val[idx]["src"] = example["text"]
            trg = "decoupled logical form" if "decoupled logical form" in example else "logical form"
            val[idx]["trg"] = example[trg]

        for idx, example in enumerate(test):
            if test_lang.startswith("hi_") or (lang is not None and lang.startswith("hi_")):
                if backtranslated:
                    test[idx]["src"] = example["back translated text"]
                else:
                    test[idx]["src"] = example["translated text"]
                    
            elif test_lang != "en":
                if backtranslated:
                    test[idx]["src"] = example["back translated text"]
                else:
                    test[idx]["src"] = example["translated text"]
                    
            else:
                test[idx]["src"] = example["text"]
            
            trg = "decoupled logical form" if "decoupled logical form" in example else "logical form"
            test[idx]["trg"] = example[trg]

    train_data = Dataset.from_pandas(pd.DataFrame(data=train))

    val_data = Dataset.from_pandas(pd.DataFrame(data=val))

    test_data = Dataset.from_pandas(pd.DataFrame(data=test))

    dataset = DatasetDict()

    dataset["train"] = train_data

    dataset["val"] = val_data

    dataset["test"] = test_data

    return dataset


def preprocess(examples, tokenizer, lang):
    lang = lang[:2]

    if "mt5" in tokenizer.name_or_path.lower():
        prefix = f"Parse from {lang} to english logical form: "
    else:
        prefix = ""

    if "indicbart" in tokenizer.name_or_path.lower():
        suffix = f" <\s> <2{lang}>"
    else:
        suffix = ""

    inputs = [prefix + example + suffix for example in examples["src"]]
    targets = [example for example in examples["trg"]]

    model_inputs = tokenizer(
        inputs,
        max_length=tokenizer.max_source_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=False if suffix else True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=tokenizer.max_target_length,
            padding="max_length",
            truncation=True,
        )

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_tokenizer(model_checkpoint, dataset_name, lang):
    lang = lang[:2]

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        cache_dir="models/",
        use_fast=False if "mt5" in model_checkpoint.lower() else True,
    )

    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"

    tokenizer.bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    tokenizer.eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    tokenizer.pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    
    added_vocab = intents_slots[dataset_name]['intents'] + intents_slots[dataset_name]['slots']
    
    if "indicbart" in model_checkpoint:
        added_vocab += [f"<2{lang}>" for lang in INDIC]
    
    tokenizer.add_tokens(added_vocab)
    

    # if "indic-atis" not in dataset_name:    
    #     tokenizer.model_max_length = 64
    # else:
    #     tokenizer.model_max_length = 32
    
    tokenizer.model_max_length = 64
        
    # Define label pad token id
    label_pad_token_id = -100
    padding = True

    if "mbart" in model_checkpoint:
        tokenizer.src_lang = mbart_dict[lang[:2]]
        tokenizer.tgt_lang = "en_XX"

    return tokenizer


def get_model(model_checkpoint, tokenizer, encoder_decoder=False):

    if encoder_decoder:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            model_checkpoint, model_checkpoint, tie_encoder_decoder=True
        )

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_checkpoint, cache_dir="models/"
        )

    if encoder_decoder:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        model.config.max_length = 64

        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 1
        model.config.length_penalty = 2.0
        model.config.repetition_penalty = 3.0
        model.config.num_beams = 10
        model.config.vocab_size = model.config.encoder.vocab_size

        # model.encoder.resize_token_embeddings(len(tokenizer))
        model.decoder.resize_token_embeddings(len(tokenizer))

    else:
        model.resize_token_embeddings(len(tokenizer))
        

    #     if "mbart" in model_checkpoint:
    #         model.config.decoder_start_token_id = tokenizer.lang_code_to_id[mbart_dict[lang[:2]]]

    #     else:
    #         model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(
    #             lang)

    return model


def prepare_dataset(
    dataset, dataset_name, tokenizer, model, train_lang="en", test_lang="en"
):
    
    tokenizer.max_target_length = 64
    tokenizer.max_source_length = 64
    model.config.max_length = 64
    
    
#     if "indic-atis" in dataset_name:
#         tokenizer.max_target_length = 64
#         tokenizer.max_source_length = 64
#         model.config.max_length = 64
    
#     else:
#         tokenizer.max_target_length = 32
#         tokenizer.max_source_length = 32
#         model.config.max_length = 32

    if "mbart" in tokenizer.name_or_path:
        tokenizer.src_lang = mbart_dict[train_lang[:2]]
        tokenizer.tgt_lang = "en_XX"
    #         model.config.decoder_start_token_id = tokenizer.lang_code_to_id[mbart_dict[train_lang[:2]]]

    #     else:
    #         model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(train_lang)

    new_dataset = DatasetDict()

    new_dataset["train"] = dataset["train"].map(
        lambda x: preprocess(x, tokenizer, train_lang),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc=f"Preprocessing {train_lang} train data",
    )

    if "mbart" in tokenizer.name_or_path:
        tokenizer.src_lang = mbart_dict[test_lang[:2]]
        tokenizer.tgt_lang = "en_XX"
    #     model.config.decoder_start_token_id = tokenizer.lang_code_to_id[mbart_dict[test_lang[:2]]]
    # else:
    #     model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(test_lang)

    new_dataset["val"] = dataset["val"].map(
        lambda x: preprocess(x, tokenizer, test_lang),
        batched=True,
        remove_columns=dataset["val"].column_names,
        desc=f"Preprocessing {test_lang} validation data",
    )

    new_dataset["test"] = dataset["test"].map(
        lambda x: preprocess(x, tokenizer, test_lang),
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc=f"Preprocessing {test_lang} test data",
    )

    if "token_type_ids" in new_dataset["train"].column_names:
        new_dataset["train"] = new_dataset["train"].remove_columns("token_type_ids")
        new_dataset["val"] = new_dataset["val"].remove_columns("token_type_ids")
        new_dataset["test"] = new_dataset["test"].remove_columns("token_type_ids")

    return new_dataset


def create_dataloaders(
    dataset,
    train_batch_size=1,
    eval_batch_size=1,
    loader_type=None,
    collate_fn=default_data_collator,
):
    if loader_type == "test":
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=collate_fn
        )
        return dataloader

    else:
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=train_batch_size,
            collate_fn=collate_fn,
        )
        val_dataloader = DataLoader(
            dataset["val"],
            shuffle=False,
            batch_size=eval_batch_size,
            collate_fn=collate_fn,
        )

        return train_dataloader, val_dataloader


# hyperparameters = {
#     "learning_rate": 1e-3,
#     "num_epochs": 1000,  # set to very high number
#     # Actual batch size will this x 8 (was 8 before but can cause OOM)
#     "train_batch_size": 16,
#     # Actual batch size will this x 8 (was 32 before but can cause OOM)
#     "eval_batch_size": 16,
#     "seed": 42,
#     "patience": 2,  # early stopping
#     "output_dir": "/content/",
#     "gradient_accumulation_steps": 4,
#     "num_warmup_steps": 500,
#     "weight_decay": 0.0,
# }

# import datasets


def train(model, tokenizer, dataset, args, hyperparameters=hyperparameters):
    # Initialize accelerator
    model_name = (
        model.name_or_path if model.name_or_path else model.encoder.name_or_path
    )

    model_parameters = model.num_parameters()

    accelerator = Accelerator(
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"]
    )

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    # Prepare everything

    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(hyperparameters["seed"])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    train_dataloader, val_dataloader = create_dataloaders(
        dataset,
        train_batch_size=hyperparameters["train_batch_size"],
        eval_batch_size=hyperparameters["eval_batch_size"],
        collate_fn=data_collator,
    )

    model = accelerator.prepare(model)

    # Instantiate optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": hyperparameters["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=hyperparameters["learning_rate"], 
    )

    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.

    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=hyperparameters["num_warmup_steps"],
    #     num_training_steps=len(train_dataloader) * hyperparameters["num_epochs"],
    # )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                                                    optimizer=optimizer,
                                                    step_size=model_step_size[model_name]
                                                    )

    (
        optimizer,
        train_dataloader,
        val_dataloader,
        # lr_scheduler,
    ) = accelerator.prepare(optimizer, 
                            train_dataloader,
                            val_dataloader, 
                            # lr_scheduler
                           )

    # Now we train the model
    epochs_no_improve = 0
    min_val_loss = 1000000

    accelerator.print(
        f"==============================================================================================="
    )
    accelerator.print(f"\t\t{model_name}\t\t\t{model_parameters//1000_000} M")
    accelerator.print(
        f"==============================================================================================="
    )

    for epoch in range(hyperparameters["num_epochs"]):
        # We only enable the progress bar on the main process to avoid having 8 progress bars.
        progress_bar = tqdm(
            range(len(train_dataloader)), disable=not accelerator.is_main_process
        )

        progress_bar.set_description(f"Epoch: {epoch}")
        model.train()

        # total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

        # Evaluate at the end of the epoch (distributed evaluation as we have 8 TPU cores)
        model.eval()
        validation_losses = []

        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss

            # We gather the loss from the 8 TPU cores to have them all.
            validation_losses.append(accelerator.gather(loss[None]))

        # Compute average validation loss
        val_loss = torch.stack(validation_losses).mean().item()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: validation loss:", val_loss)

        progress_bar.update(1)
        progress_bar.set_postfix(
            {"epoch": epoch, "val_loss": val_loss},
        )

        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            # output_dir = f"val_loss_{val_loss}"
            # output_dir = os.path.join(hyperparameters["output_dir"], output_dir)
            # accelerator.save_state(output_dir)
            continue

        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == hyperparameters["patience"]:
                accelerator.print("Early stopping!")
                break

    # save trained model
    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # Use accelerator.save to save
    # unwrapped_model.save_pretrained(hyperparameters["output_dir"], save_function=accelerator.save)

    # model = accelerator.unwrap_model(model)


base_path = "results"


def make_pth(strategy, dataset, model, lang=None):

    if lang:
        path = f"{base_path}/{strategy}/{dataset}/{model}/{lang}"
    else:
        path = f"{base_path}/{strategy}/{dataset}/{model}"

    return path


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def tree_labelled_f1(predictions, ground_truth):
    return np.mean([res['tree_labeled_bracketing_scores']['f1'] for res in evaluate_predictions(predictions, ground_truth)], axis=0)


def evaluate(gold_answers, predictions, tokenizer):
    scores = []
    exact_matches = []
    lev_dist = []

    for ground_truth, prediction in zip(gold_answers, predictions):

        exact_match_metric = exact_match([prediction.lower()], [ground_truth.lower()], tokenizer)
        f1 = tree_labelled_f1([prediction.lower()], [ground_truth.lower()])

        dist = lev.ratio(prediction.lower(), ground_truth.lower())

        exact_matches.append(exact_match_metric)
        scores.append(f1)
        lev_dist.append(dist)

    return {
        "exact_match": exact_matches,
        "tree_labelled_f1": scores,
        "levenshtein_distance": lev_dist,
    }


def exact_match(decoded_preds, decoded_labels, tokenizer):

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        regexes_to_ignore=[r"\s+"],
        ignore_case=True,
        ignore_punctuation=True,
    )
    result = {"exact_match": result["exact_match"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in decoded_preds
    ]

    result = {k: round(v, 4) for k, v in result.items()}
    return round(result["exact_match"], 1)



def post_process_lf(sent):
    stck = 0
    res = []
    
    toks = sent.split()
    
    
    for tok in toks:
        
        if tok.startswith("["):
            stck += 1
        elif tok == ']':
            stck -= 1
        
        res.append(tok)
        
        if stck == 0:
            break
        
        
    return ' '.join(res) 





def generate(
    model,
    tokenizer,
    dataset,
    raw_dataset,
    technique,
    dataset_name,
    lang,
    hyperparameters=hyperparameters,
):

    model_name = (
        model.name_or_path if model.name_or_path else model.encoder.name_or_path
    )

    accelerator = Accelerator()

    accelerator.print("generating predictions ........")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    data_loader = create_dataloaders(
        dataset, eval_batch_size=8, loader_type="test", collate_fn=data_collator
    )

    model, data_loader = accelerator.prepare(model, data_loader)

    # this will set a different random seed per device

    torch.manual_seed(random.randrange(0, 10000))
    preds = []
    labels = []

    progress_bar = tqdm(
        range(len(data_loader)), disable=not accelerator.is_main_process
    )

    samples_seen = 0
    for step, batch in enumerate(data_loader):
        with torch.no_grad():

            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=3,
                # max_length=64 if dataset_name == "indic-atis" else 32,
                max_length=64,
                use_cache=True,
                early_stopping=True
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            label_batch = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )
            label_batch = accelerator.gather(label_batch).cpu().numpy()

            label_batch = np.where(
                label_batch != -100, label_batch, tokenizer.pad_token_id
            )
            decoded_labels = tokenizer.batch_decode(
                label_batch, skip_special_tokens=True
            )

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )
            if accelerator.num_processes > 1:
                if step == len(data_loader) - 1:
                    decoded_preds = decoded_preds[
                        : len(data_loader.dataset) - samples_seen
                    ]
                    decoded_labels = decoded_labels[
                        : len(data_loader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += len(decoded_labels)
            preds += decoded_preds

            labels += decoded_labels

        progress_bar.update(1)
        progress_bar.set_postfix({"steps": step})

    with accelerator.main_process_first():
        model_name = model_name.split("/")[-1] if "/" in model_name else model_name

        if technique == "crosslingual_transfer":
            pth = make_pth(technique, dataset_name, model_name, lang)

        else:
            pth = make_pth(technique, dataset_name, model_name)

        save_data = {k: raw_dataset[k] for k in raw_dataset.column_names}
        
        
        save_data["predictions"] = preds
        save_data['post_processed_predictions'] = list(map(post_process_lf, preds))
        save_data["labels"] = labels

        save_data["trg"], _ = postprocess_text(save_data["trg"], [])

        metrics = evaluate(save_data["labels"], save_data["post_processed_predictions"], tokenizer)

        for k, v in metrics.items():
            save_data[k] = v

        save_dict = pd.DataFrame(save_data).to_dict("records")

        with open(f"{pth}/{lang}.json", "w", encoding="utf-8") as f:
            json.dump({"outputs": save_dict}, f, indent=6, ensure_ascii=False)
    
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    


def remove_model():
    files = glob.glob("models/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="If passed, will train on the CPU."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    args = parser.parse_args()
    return args
