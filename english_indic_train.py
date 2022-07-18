from utils import *


# """# Define Training Parameters"""

seq2seq_models = [ 
                  'ai4bharat/IndicBART',
                  'google/mt5-base', 
                  "facebook/mbart-large-50",
                  "facebook/mbart-large-50-many-to-one-mmt",
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
    "output_dir": "trained_model/",
    "gradient_accumulation_steps": 4,
    "num_warmup_steps": 0,
    "weight_decay": 0.0
}


batch_sizes_gpu = {
                  'ai4bharat/IndicBART': 320,
                  'google/mt5-base': 64, 
                  "facebook/mbart-large-50": 64,
                  "facebook/mbart-large-50-many-to-one-mmt": 64,
                  'xlm-roberta-base': 32,
                  "google/muril-base-cased": 36
}

model_lr = {
    'ai4bharat/IndicBART': 1e-3,
    'google/mt5-base': 1e-3,
    "facebook/mbart-large-50": 1e-3,
    "facebook/mbart-large-50-many-to-one-mmt": 1e-3,
    'xlm-roberta-base': 3e-5,
    "google/muril-base-cased":3e-5
            
}

model_epochs_gpu = {
                  'ai4bharat/IndicBART': 10,
                  'google/mt5-base': 8, 
                  "facebook/mbart-large-50": 8,
                  "facebook/mbart-large-50-many-to-one-mmt": 8,
                  'xlm-roberta-base': 5,
                  "google/muril-base-cased": 5
}


def main():
    args = get_args()
    if "english_indic_train" not in os.listdir(base_path):
        os.mkdir(os.path.join(base_path, "english_indic_train"))

    for dataset_name in dataset_names:
        print(f"dataset:{dataset_name}")

        if dataset_name not in os.listdir(f"{base_path}/english_indic_train/"):
            os.mkdir(f"{base_path}/english_indic_train/{dataset_name}")

        for model_checkpoint in list(seq2seq_models + encoder_models):
            model_name = model_checkpoint.split(
                "/")[-1] if '/' in model_checkpoint else model_checkpoint

            print(f"model:{model_name}")

            if model_name not in os.listdir(f"{base_path}/english_indic_train/{dataset_name}/"):
                os.mkdir(
                    f"{base_path}/english_indic_train/{dataset_name}/{model_name}")

            for lang in INDIC:
                print(f"language:{lang}")

                if f"{lang}.json" in os.listdir(f"{base_path}/english_indic_train/{dataset_name}/{model_name}/"):
                    print("Skipping.......")
                    continue

                raw_dataset = create_dataset(dataset_name, "en", "en")
                # print(raw_dataset)

                tokenizer = get_tokenizer(model_checkpoint, "en")

                if model_checkpoint in encoder_models:
                    model = get_model(model_checkpoint,
                                      tokenizer, lang, encoder_decoder=True)

                else:
                    model = get_model(model_checkpoint, tokenizer, lang)

                english_dataset = prepare_dataset(raw_dataset, dataset_name, tokenizer, "en", "en")

                hyperparameters['train_batch_size'] = batch_sizes_gpu[model_checkpoint]
                hyperparameters['eval_batch_size'] = batch_sizes_gpu[model_checkpoint]
                hyperparameters['num_epochs'] = model_epochs_gpu[model_checkpoint]
                hyperparameters["learning_rate"] = model_lr[model_checkpoint]


                train(
                        model, 
                        tokenizer, 
                        english_dataset,
                        args, 
                        hyperparameters
                    )

                tokenizer = get_tokenizer(model_checkpoint, lang, lang)
                
                raw_dataset = create_dataset(dataset_name, lang, lang)

                dataset = prepare_dataset(
                    raw_dataset, dataset_name, tokenizer, lang, lang)

                train(
                        model, 
                        tokenizer, 
                        dataset,
                        args, 
                        hyperparameters
                        )

                generate(
                            model, 
                            tokenizer, 
                            dataset['test'], 
                            raw_dataset['test'], 
                            "english_indic_train",
                            dataset_name, 
                            lang, 
                            hyperparameters
                        )

            remove_model()


if __name__ == "__main__":
    main()
