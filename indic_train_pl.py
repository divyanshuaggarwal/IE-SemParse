
import torch
import gc
import os
import json
import pandas as pd
import numpy as np

from utils_pl import *

seq2seq_models = [
    'ai4bharat/IndicBART',
    'google/mt5-base',
    "facebook/mbart-large-50",
]

encoder_models = [
    # 'xlm-roberta-base',
    # "google/muril-base-cased"
]

dataset_names = ["itop", "indic-TOP", "indic-atis"]

INDIC = ['hi', 'bn', 'mr', 'as', 'ta', 'te', 'or', 'ml', 'pa', 'gu', 'kn']

seed_everything(42)

technique = "indic_train"

for dataset_name in dataset_names:
    if "indic_train" not in os.listdir(base_path):
        os.mkdir(os.path.join(base_path, "indic_train"))

    print(f"dataset:{dataset_name}")

    if dataset_name not in os.listdir(f"{base_path}/{technique}/"):
        os.mkdir(f"{base_path}/{technique}/{dataset_name}")

    for model_checkpoint in list(seq2seq_models + encoder_models):

        model_name = model_checkpoint.split(
            "/")[-1] if '/' in model_checkpoint else model_checkpoint

        print(f"model:{model_name}")

        if model_name not in os.listdir(f"{base_path}/{technique}/{dataset_name}/"):
            os.mkdir(f"{base_path}/{technique}/{dataset_name}/{model_name}")

        for lang in INDIC:
            print(f"language:{lang}")

            if f"{lang}.json" in os.listdir(f"{base_path}/{technique}/{dataset_name}/{model_name}/"):
                print("Skipping.......")
                continue

            tokenizer = get_tokenizer(model_checkpoint, lang)

            if model_checkpoint in encoder_models:
                model = get_model(model_checkpoint, tokenizer,
                                  lang, encoder_decoder=True)

            else:
                model = get_model(model_checkpoint, tokenizer, lang)

            dm = SemanticParseDataModule(
                model_name=model_checkpoint,
                dataset=dataset_name,
                train_lang=lang,
                test_lang=lang,
                tokenizer=tokenizer,
                model=model,

            )

            pl_model = Parser(
                model_name=model_checkpoint,
                dataset=dataset_name,
                lang=lang,
                technique=technique,
                tokenizer=tokenizer,
                model=model,
            )

            pl_model, dm = tune(pl_model, dm)

            trainer = get_trainer()
            # trainer.tune(pl_model, dm)

            gc.collect()
            torch.cuda.empty_cache()

            trainer.fit(pl_model, dm)

            # trainer = Trainer(
            #                   precision=16,
            #                   accelerator = "auto",
            #                   devices = "auto"
            #                   )

            trainer.test(pl_model, dm)
