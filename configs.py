# """# Define Training Parameters"""

import argparse

seq2seq_models = [
    'ai4bharat/IndicBART',
    'google/mt5-base',
    "facebook/mbart-large-50",
    "facebook/mbart-large-50-many-to-one-mmt",
    "ai4bharat/IndicBART-XXEN",
]

encoder_models = [
    #   'xlm-roberta-base',
    #   "google/muril-base-cased"
]

dataset_names = [
                    # "itop", 
                    # "indic-TOP", 
                    "indic-atis"
]

INDIC = ['en', 'hi', 'bn', 'mr', 'as', 'ta', 'te', 'or', 'ml', 'pa', 'gu', 'kn']

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
    "weight_decay": 0.0
}


batch_sizes_gpu = {
    'ai4bharat/IndicBART': 128,
    'google/mt5-base': 32,
    "facebook/mbart-large-50": 32,
    "facebook/mbart-large-50-many-to-one-mmt": 32,
    'xlm-roberta-base': 32,
    "google/muril-base-cased": 36
}

model_lr = {
    'ai4bharat/IndicBART': 1e-3,
    'google/mt5-base': 1e-3,
    "facebook/mbart-large-50": 1e-4,
    "facebook/mbart-large-50-many-to-one-mmt": 1e-4,
    'xlm-roberta-base': 3e-5,
    "google/muril-base-cased": 3e-5

}

model_epochs_gpu = {
    'ai4bharat/IndicBART': 20,
    'google/mt5-base': 10,
    "facebook/mbart-large-50": 8,
    "facebook/mbart-large-50-many-to-one-mmt": 8,
    'xlm-roberta-base': 5,
    "google/muril-base-cased": 5
}

def get_args():
    parser = argparse.ArgumentParser(
        description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true",
                        help="If passed, will train on the CPU.")
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
