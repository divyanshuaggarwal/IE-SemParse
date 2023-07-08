from utils import *
from configs import *
from datasets import concatenate_datasets, DatasetDict


def make_combined_dataset(dataset_name, lang, model_checkpoint):
    datasets = []
    for lang in ['en', lang]:
        raw_dataset = create_dataset(dataset_name, lang, lang)
        tokenizer = get_tokenizer(model_checkpoint, dataset_name, lang)

        if model_checkpoint in encoder_models:
            model = get_model(model_checkpoint, tokenizer, lang, encoder_decoder=True)

        else:
            model = get_model(model_checkpoint, tokenizer)

        dataset = prepare_dataset(
            raw_dataset, dataset_name, tokenizer, model, lang, lang
        )
        datasets.append(dataset)

    final_dataset = DatasetDict()

    final_dataset["train"] = concatenate_datasets(
        [dataset["train"] for dataset in datasets]
    )
    final_dataset["val"] = concatenate_datasets(
        [dataset["val"] for dataset in datasets]
    )
    final_dataset["test"] = concatenate_datasets(
        [dataset["test"] for dataset in datasets]
    )

    return final_dataset


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
                if f"{lang}.json" not in os.listdir(f"/workspace/Indic-SemParse/Indic-SemParse/filtered_data/{dataset_name}/"):
                    continue
                    
                print(f"language:{lang}")

                if f"{lang}.json" in os.listdir(
                    f"{base_path}/english_indic_train/{dataset_name}/{model_name}/"
                ):
                    print("Skipping.......")
                    continue

                # raw_dataset = create_dataset(dataset_name, "en", "en")
                # # print(raw_dataset)

                # tokenizer = get_tokenizer(model_checkpoint, dataset_name, "en")

                

                # english_dataset = prepare_dataset(
                #     raw_dataset, dataset_name, tokenizer, "en", "en"
                # )

                hyperparameters["train_batch_size"] = batch_sizes_gpu[model_checkpoint]
                hyperparameters["eval_batch_size"] = batch_sizes_gpu[model_checkpoint]
                hyperparameters["num_epochs"] = model_epochs_gpu[model_checkpoint]
                hyperparameters["learning_rate"] = model_lr[model_checkpoint]

                # train(model, tokenizer, english_dataset, args, hyperparameters)

                tokenizer = get_tokenizer(model_checkpoint, dataset_name, lang)
                
                if model_checkpoint in encoder_models:
                    model = get_model(model_checkpoint, tokenizer, encoder_decoder=True)

                else:
                    model = get_model(model_checkpoint, tokenizer)

                raw_dataset = create_dataset(dataset_name, lang, lang)

                lang_dataset = prepare_dataset(
                    raw_dataset, dataset_name, tokenizer, model, lang, lang
                )

                dataset = make_combined_dataset(dataset_name, lang, model_checkpoint)

                train(model, tokenizer, dataset, args, hyperparameters)

                generate(
                    model,
                    tokenizer,
                    lang_dataset["test"],
                    raw_dataset["test"],
                    "english_indic_train",
                    dataset_name,
                    lang,
                    hyperparameters,
                )

            remove_model()


if __name__ == "__main__":
    main()
