from pprint import pprint
from utils import *
from configs import *


english_models = [
                  't5-base', 
                  't5-large', 
                  'facebook/bart-large',
                  'facebook/bart-base'
                 ]

def main():
    args = get_args()
    if "translate_test" not in os.listdir(base_path):
        os.mkdir(os.path.join(base_path, "translate_test"))

    for dataset_name in dataset_names:
        print(f"dataset:{dataset_name}")

        if dataset_name not in os.listdir(f"{base_path}/translate_test/"):
            os.mkdir(f"{base_path}/translate_test/{dataset_name}")

        for model_checkpoint in list(
            seq2seq_models + encoder_models + translation_models + english_models
        ): 
            model_name = (
                model_checkpoint.split("/")[-1]
                if "/" in model_checkpoint
                else model_checkpoint
            )

            print(f"model:{model_name}")

            if model_name in os.listdir(f"{base_path}/translate_test/{dataset_name}/"):
                if len(
                    os.listdir(
                        f"{base_path}/translate_test/{dataset_name}/{model_name}"
                    )
                ) == len(INDIC):
                    continue

            else:
                os.mkdir(f"{base_path}/translate_test/{dataset_name}/{model_name}")
            
            print(len(os.listdir(f"/workspace/Indic-SemParse/results/translate_test/{dataset_name}/{model_name}/")))
            print(len(INDIC))
            print(len(os.listdir(f"/workspace/Indic-SemParse/results/translate_test/{dataset_name}/{model_name}/")) >= len(INDIC))
            
            if dataset_name == "indic-TOP":
                if len(os.listdir(f"/workspace/Indic-SemParse/results/translate_test/{dataset_name}/{model_name}/")) >= len(INDIC)-3:
                    print("skipping ....")
                    continue
                
            else:
                if len(os.listdir(f"/workspace/Indic-SemParse/results/translate_test/{dataset_name}/{model_name}/")) >= len(INDIC)-2:
                    print("skipping ....")
                    continue


            for lang in INDIC:
                
                if lang in ("hi_mtop", "hi_atis"):
                    print("skipping ....")
                    continue
                
                if dataset_name != "Indic-TOP" and lang == "en":
                    continue
                
                if dataset_name == "Indic-TOP" and lang == "hi_orig":
                    continue
                
                if lang != "en":
                    if f"{lang}.json" in os.listdir(f"/workspace/Indic-SemParse/Indic-SemParse/filtered_data/{dataset_name}/"):
                        continue
                
                
                print(f"language:{lang}")
                
                if f"{lang}.json" in os.listdir(f"/workspace/Indic-SemParse/results/translate_test/{dataset_name}/{model_name}/"):
                    print("skipping ....")
                    continue
                
                
                if dataset_name == "indic-TOP":
                    raw_dataset = create_dataset(dataset_name, "en", "en", None, False, True)
                else:
                    raw_dataset = create_dataset(dataset_name, "en", "en", lang)
                
                pprint(raw_dataset['train'][0])
                pprint(raw_dataset['val'][0])

                # print(raw_dataset['train'][0])
                # print(raw_dataset['val'][0])
                # print(raw_dataset['test'][0])

                tokenizer = get_tokenizer(model_checkpoint, dataset_name, "en")

                if model_checkpoint in encoder_models:
                    model = get_model(model_checkpoint, tokenizer, encoder_decoder=True)

                else:
                    model = get_model(model_checkpoint, tokenizer)

                dataset = prepare_dataset(
                    raw_dataset, dataset_name, tokenizer, model, "en", "en"
                )

                hyperparameters["train_batch_size"] = batch_sizes_gpu[model_checkpoint]
                hyperparameters["eval_batch_size"] = batch_sizes_gpu[model_checkpoint]
                hyperparameters["num_epochs"] = model_epochs_gpu[model_checkpoint]
                hyperparameters["learning_rate"] = model_lr[model_checkpoint]

                pprint(hyperparameters)    
            
                train(model, tokenizer, dataset, args, hyperparameters)
    

                if f"{lang}.json" in os.listdir(
                    f"{base_path}/translate_test/{dataset_name}/{model_name}/"
                ):
                    print("Skipping.......")
                    continue
                    
                if dataset_name == "indic-TOP":
                    raw_dataset = create_dataset(dataset_name, lang, lang, None, False, True)
                else:
                    raw_dataset = create_dataset(dataset_name, lang, lang, None, True)
                
                pprint(raw_dataset['test'][0])

                dataset = prepare_dataset(
                    raw_dataset, dataset_name, tokenizer, model, "en", "en"
                )

                # print(raw_dataset['train'][0])
                # print(raw_dataset['val'][0])
                # print(raw_dataset['test'][0])

                generate(
                    model,
                    tokenizer,
                    dataset["test"],
                    raw_dataset["test"],
                    "translate_test",
                    dataset_name,
                    lang,
                )

            remove_model()


if __name__ == "__main__":
    main()
