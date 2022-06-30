from utils import *


def get_args():
    parser = argparse.ArgumentParser(
        description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
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

                raw_dataset = create_dataset(dataset_name, lang)

                tokenizer = get_tokenizer(model_checkpoint, lang)

                if model_checkpoint in encoder_models:
                    model = get_model(model_checkpoint,
                                      tokenizer, lang, encoder_decoder=True)

                else:
                    model = get_model(model_checkpoint, tokenizer, lang)

                dataset = prepare_dataset(raw_dataset, tokenizer)

                # def _train():
                training_function(model, tokenizer, dataset,
                                  args, hyperparameters)

                # notebook_launcher(_train, use_fp16 = True)

                # def _generate():
                generate(
                    model, tokenizer, dataset['test'], raw_dataset['test'], "indic_train", dataset_name, lang)

                # notebook_launcher(_generate, use_fp16 = True)


if __name__ == "__main__":
    main()
