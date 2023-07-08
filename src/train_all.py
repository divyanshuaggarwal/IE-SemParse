from pprint import pprint
from utils import *
from datasets import concatenate_datasets, DatasetDict
from configs import *


INDIC = [
    "hi_orig",
    "hi_atis",
    "hi_mtop",
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

def make_combined_dataset(dataset_name, model_checkpoint):
    datasets = []
    for lang in ['en'] + INDIC:
        if f"{lang}.json" not in os.listdir(f"/workspace/Indic-SemParse/Indic-SemParse/filtered_data/{dataset_name}/"):
                    continue
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
    if "train_all" not in os.listdir(base_path):
        os.mkdir(os.path.join(base_path, "train_all"))
    
    print(dataset_names)

    for dataset_name in dataset_names:
        print(f"dataset:{dataset_name}")

        if dataset_name not in os.listdir(f"{base_path}/train_all/"):
            os.mkdir(f"{base_path}/train_all/{dataset_name}")

        for model_checkpoint in list(
            seq2seq_models + encoder_models + translation_models
        ):
            model_name = (
                model_checkpoint.split("/")[-1]
                if "/" in model_checkpoint
                else model_checkpoint
            )

            print(f"model:{model_name}")

            if model_name in os.listdir(f"{base_path}/train_all/{dataset_name}/"):
                if len(
                    os.listdir(f"{base_path}/train_all/{dataset_name}/{model_name}")
                ) >= len(INDIC) - 1:
                    print("skipping....")
                    continue

            else:
                os.mkdir(f"{base_path}/train_all/{dataset_name}/{model_name}")

            # print(raw_dataset['train'][0])
            # print(raw_dataset['val'][0])
            # print(raw_dataset['test'][0])

            tokenizer = get_tokenizer(model_checkpoint, dataset_name, "en")

            if model_checkpoint in encoder_models:
                model = get_model(model_checkpoint, tokenizer, encoder_decoder=True)

            else:
                model = get_model(model_checkpoint, tokenizer)

            dataset = make_combined_dataset(dataset_name, model_checkpoint)

            hyperparameters["train_batch_size"] = batch_sizes_gpu[model_checkpoint]
            hyperparameters["eval_batch_size"] = batch_sizes_gpu[model_checkpoint]
            hyperparameters["num_epochs"] = model_epochs_gpu[model_checkpoint]
            hyperparameters["learning_rate"] = model_lr[model_checkpoint]
            hyperparameters["patience"] = model_patience[model_checkpoint]

            pprint(hyperparameters)
            train(model, tokenizer, dataset, args, hyperparameters)
            

            for lang in INDIC:
                if f"{lang}.json" not in os.listdir(f"/workspace/Indic-SemParse/Indic-SemParse/unfiltered_data/{dataset_name}/"):
                    continue
                    
                print(f"language:{lang}")

                if f"{lang}.json" in os.listdir(
                    f"{base_path}/train_all/{dataset_name}/{model_name}/"
                ):
                    print("Skipping.......")
                    continue

                raw_dataset = create_dataset(dataset_name, lang, lang)

                dataset = prepare_dataset(
                    raw_dataset, dataset_name, tokenizer, model, lang, lang
                )

                # print(raw_dataset['train'][0])
                # print(raw_dataset['val'][0])
                # print(raw_dataset['test'][0])

                generate(
                    model,
                    tokenizer,
                    dataset["test"],
                    raw_dataset["test"],
                    "train_all",
                    dataset_name,
                    lang,
                )

            remove_model()


if __name__ == "__main__":
    main()