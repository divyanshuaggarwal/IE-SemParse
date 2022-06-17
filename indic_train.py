from utils import *


def main():
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
                training_function(model, tokenizer, dataset, hyperparameters)

                # notebook_launcher(_train, use_fp16 = True)

                # def _generate():
                generate(
                    model, tokenizer, dataset['test'], raw_dataset['test'], "indic_train", dataset_name, lang)

                # notebook_launcher(_generate, use_fp16 = True)


if __name__ == "__main__":
    main()
