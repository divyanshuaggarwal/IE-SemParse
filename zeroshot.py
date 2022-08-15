from utils import *
from pprint import pprint
from configs import *




def main():
    args = get_args()
    if "english_train" not in os.listdir(base_path):
        os.mkdir(os.path.join(base_path, "english_train"))

    for dataset_name in dataset_names:
        print(f"dataset:{dataset_name}")

        if dataset_name not in os.listdir(f"{base_path}/english_train/"):
            os.mkdir(f"{base_path}/english_train/{dataset_name}")

        for model_checkpoint in list(seq2seq_models + encoder_models):
            model_name = model_checkpoint.split(
                "/")[-1] if '/' in model_checkpoint else model_checkpoint

            print(f"model:{model_name}")

            if model_name not in os.listdir(f"{base_path}/english_train/{dataset_name}/"):
                os.mkdir(
                    f"{base_path}/english_train/{dataset_name}/{model_name}")

            for lang in INDIC:
                print(f"language:{lang}")

                if f"{lang}.json" in os.listdir(f"{base_path}/english_train/{dataset_name}/{model_name}/"):
                    print("Skipping.......")
                    continue

                raw_dataset = create_dataset(dataset_name, "en", lang)

                # pprint(raw_dataset['train'][0])
                # pprint(raw_dataset['val'][0])
                # pprint(raw_dataset['test'][0])

                tokenizer = get_tokenizer(model_checkpoint, "en")

                if model_checkpoint in encoder_models:
                    model = get_model(model_checkpoint, tokenizer, lang, encoder_decoder=True)

                else:
                    model = get_model(model_checkpoint, tokenizer, lang)

                dataset = prepare_dataset(raw_dataset, dataset_name, tokenizer, model, "en", lang)
                
                hyperparameters['train_batch_size'] = batch_sizes_gpu[model_checkpoint] 
                hyperparameters['eval_batch_size'] = batch_sizes_gpu[model_checkpoint] 
                hyperparameters['num_epochs'] = model_epochs_gpu[model_checkpoint]
                hyperparameters["learning_rate"] = model_lr[model_checkpoint]

                train(model, tokenizer, dataset, args, hyperparameters)

                # notebook_launcher(_train, use_fp16 = True)

                # def _generate():
                generate(model, tokenizer, dataset['test'], raw_dataset['test'], "english_train", dataset_name, lang)

                # notebook_launcher(_generate, use_fp16 = True)
            remove_model()


if __name__ == "__main__":
    main()
