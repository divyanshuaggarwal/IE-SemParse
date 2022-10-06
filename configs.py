# """# Define Training Parameters"""

import argparse

seq2seq_models = [
    "ai4bharat/IndicBART",
    "google/mt5-base",
    "facebook/mbart-large-50",
]

translation_models = [
    "facebook/mbart-large-50-many-to-one-mmt",
    "ai4bharat/IndicBART-XXEN",
]

encoder_models = [
    #   'xlm-roberta-base',
    #   "google/muril-base-cased"
]

dataset_names = ["itop", "indic-TOP", "indic-atis"]

INDIC = [
    # 'en',
    "hi",
    "bn",
    "mr",
    "as",
    "ta",
    "te",
    "or",
    "ml",
    "pa",
    "gu",
    "kn",
]

hyperparameters = {
    "learning_rate": 1e-3,
    # "num_epochs": 1000,
    "num_epochs": 1,  # set to very high number
    # Actual batch size will this x 8 (was 8 before but can cause OOM)
    "train_batch_size": 8,
    # Actual batch size will this x 8 (was 32 before but can cause OOM)
    "eval_batch_size": 8,
    "seed": 42,
    "patience": 1,  # early stopping
    "output_dir": "trained_model/",
    "gradient_accumulation_steps": 4,
    "num_warmup_steps": 0,
    "weight_decay": 0.0,
    'early_stopping_margin': 0.1,
}

model_step_size = {
    "ai4bharat/IndicBART": 4,
    "ai4bharat/IndicBART-XXEN": 4,
    "google/mt5-base": 2,
    "facebook/mbart-large-50": 2,
    "facebook/mbart-large-50-many-to-one-mmt": 2,
    "xlm-roberta-base": 1,
    "google/muril-base-cased": 1,
}


batch_sizes_gpu = {
    "ai4bharat/IndicBART": 128,
    "ai4bharat/IndicBART-XXEN": 128,
    "google/mt5-base": 18,
    "facebook/mbart-large-50": 22,
    "facebook/mbart-large-50-many-to-one-mmt": 22,
    "xlm-roberta-base": 28,
    "google/muril-base-cased": 36,
}

model_lr = {
    "ai4bharat/IndicBART": 3e-3,
    "ai4bharat/IndicBART-XXEN": 3e-3,
    "google/mt5-base": 3e-4,
    "facebook/mbart-large-50": 1e-4,
    "facebook/mbart-large-50-many-to-one-mmt": 1e-4,
    "xlm-roberta-base": 3e-5,
    "google/muril-base-cased": 3e-5,
}

model_epochs_gpu = {
    "ai4bharat/IndicBART": 25,
    "ai4bharat/IndicBART-XXEN": 25,
    "google/mt5-base": 15,
    "facebook/mbart-large-50": 15,
    "facebook/mbart-large-50-many-to-one-mmt": 15,
    "xlm-roberta-base": 5,
    "google/muril-base-cased": 5,
}

model_patience = {
    "ai4bharat/IndicBART": 1,
    "ai4bharat/IndicBART-XXEN": 1,
    "google/mt5-base": 1,
    "facebook/mbart-large-50": 1,
    "facebook/mbart-large-50-many-to-one-mmt": 1,
    "xlm-roberta-base": 1,
    "google/muril-base-cased": 1,
}


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


