from utils import *


# """# Define Training Parameters"""

seq2seq_models = [ 
                #   'ai4bharat/IndicBART',
                #   'google/mt5-base', 
                  "facebook/mbart-large-50",
                  ]

encoder_models = [
                #   'xlm-roberta-base',
                #   "google/muril-base-cased"
                  ]

dataset_names = ["itop", "indic-TOP", "indic-atis"]

INDIC = ['hi','bn','mr','as','ta','te','or','ml','pa','gu','kn']

hyperparameters = {
    "learning_rate": 1e-3,
    # "num_epochs": 1000,
    "num_epochs": 1, # set to very high number
    "train_batch_size": 8, # Actual batch size will this x 8 (was 8 before but can cause OOM)
    "eval_batch_size": 8, # Actual batch size will this x 8 (was 32 before but can cause OOM)
    "seed": 42,
    "patience": 1, # early stopping
    "output_dir": "/content/trained_model",
    "gradient_accumulation_steps": 4,
    "num_warmup_steps": 0,
    "weight_decay": 0.0
}


batch_sizes_gpu = {
                  'ai4bharat/IndicBART': 128,
                  'google/mt5-base': 64, 
                  "facebook/mbart-large-50": 32,
                  'xlm-roberta-base': 32,
                  "google/muril-base-cased": 32
}

model_lr = {
    'ai4bharat/IndicBART': 1e-3,
    'google/mt5-base': 1e-3,
    "facebook/mbart-large-50": 1e-3,
    'xlm-roberta-base': 3e-4,
    "google/muril-base-cased":3e-4
            
}

model_epochs_gpu = {
                  'ai4bharat/IndicBART': 10,
                  'google/mt5-base': 10, 
                  "facebook/mbart-large-50": 5,
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


def main():
    args = get_args()
    if "indic_train" not in os.listdir(base_path):
        os.mkdir(os.path.join(base_path, "indic_train"))

    for dataset_name in dataset_names:
        print(f"dataset:{dataset_name}")

        if dataset_name not in os.listdir(f"{base_path}/indic_train/"):
            os.mkdir(f"{base_path}/indic_train/{dataset_name}")

        for model_checkpoint in list(seq2seq_models + encoder_models):
            model_name = model_checkpoint.split(
                "/")[-1] if '/' in model_checkpoint else model_checkpoint

            print(f"model:{model_name}")

            if model_name not in os.listdir(f"{base_path}/indic_train/{dataset_name}/"):
                os.mkdir(
                    f"{base_path}/indic_train/{dataset_name}/{model_name}")

            for lang in INDIC:
                print(f"language:{lang}")

                if f"{lang}.json" in os.listdir(f"{base_path}/indic_train/{dataset_name}/{model_name}/"):
                    print("Skipping.......")
                    continue

                raw_dataset = create_dataset(dataset_name, lang, lang)

                tokenizer = get_tokenizer(model_checkpoint, lang)

                if model_checkpoint in encoder_models:
                    model = get_model(model_checkpoint,
                                      tokenizer, lang, encoder_decoder=True)

                else:
                    model = get_model(model_checkpoint, tokenizer, lang)

                dataset = prepare_dataset(raw_dataset, tokenizer)
                hyperparameters['train_batch_size'] = batch_sizes_gpu[model_checkpoint] if torch.cuda.is_available() else batch_sizes_tpu[model_checkpoint]
                hyperparameters['eval_batch_size'] = batch_sizes_gpu[model_checkpoint] if torch.cuda.is_available() else batch_sizes_tpu[model_checkpoint]
                hyperparameters['num_epochs'] = model_epochs_gpu[model_checkpoint] if torch.cuda.is_available() else model_epochs_tpu[model_checkpoint]
                hyperparameters["learning_rate"] = model_lr[model_checkpoint]
                # def _train():
                training_function(model, tokenizer, dataset,
                                  args, hyperparameters)

                # notebook_launcher(_train, use_fp16 = True)

                # def _generate():
                generate(
                    model, tokenizer, dataset['test'], raw_dataset['test'], "indic_train", dataset_name, lang)

                # notebook_launcher(_generate, use_fp16 = True)
                remove_model()


if __name__ == "__main__":
    main()
