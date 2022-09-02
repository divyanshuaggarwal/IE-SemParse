from utils import *
from configs import *


def main():
    args = get_args()
    if "english_indic_train" not in os.listdir(base_path):
        os.mkdir(os.path.join(base_path, "english_indic_train"))

    for dataset_name in dataset_names:
        print(f"dataset:{dataset_name}")

        if dataset_name not in os.listdir(f"{base_path}/english_indic_train/"):
            os.mkdir(f"{base_path}/english_indic_train/{dataset_name}")

        for model_checkpoint in list(
            seq2seq_models + encoder_models + translation_models
        ):
            model_name = (
                model_checkpoint.split("/")[-1]
                if "/" in model_checkpoint
                else model_checkpoint
            )

            print(f"model:{model_name}")

            if model_name not in os.listdir(
                f"{base_path}/english_indic_train/{dataset_name}/"
            ):
                os.mkdir(f"{base_path}/english_indic_train/{dataset_name}/{model_name}")

            for lang in INDIC:
                print(f"language:{lang}")

                if f"{lang}.json" in os.listdir(
                    f"{base_path}/english_indic_train/{dataset_name}/{model_name}/"
                ):
                    print("Skipping.......")
                    continue

                raw_dataset = create_dataset(dataset_name, "en", "en")
                # print(raw_dataset)

                tokenizer = get_tokenizer(model_checkpoint, "en")

                if model_checkpoint in encoder_models:
                    model = get_model(model_checkpoint, tokenizer, encoder_decoder=True)

                else:
                    model = get_model(model_checkpoint, tokenizer)

                english_dataset = prepare_dataset(
                    raw_dataset, dataset_name, tokenizer, "en", "en"
                )

                hyperparameters["train_batch_size"] = batch_sizes_gpu[model_checkpoint]
                hyperparameters["eval_batch_size"] = batch_sizes_gpu[model_checkpoint]
                hyperparameters["num_epochs"] = model_epochs_gpu[model_checkpoint]
                hyperparameters["learning_rate"] = model_lr[model_checkpoint]

                train(model, tokenizer, english_dataset, args, hyperparameters)

                tokenizer = get_tokenizer(model_checkpoint, lang)

                raw_dataset = create_dataset(dataset_name, lang, lang)

                dataset = prepare_dataset(
                    raw_dataset, dataset_name, tokenizer, lang, lang
                )

                train(model, tokenizer, dataset, args, hyperparameters)

                generate(
                    model,
                    tokenizer,
                    dataset["test"],
                    raw_dataset["test"],
                    "english_indic_train",
                    dataset_name,
                    lang,
                    hyperparameters,
                )

            # remove_model()


if __name__ == "__main__":
    main()
