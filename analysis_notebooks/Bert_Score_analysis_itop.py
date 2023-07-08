#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


# !wget https://ai4b-my.sharepoint.com/:u:/g/personal/sumanthdoddapaneni_ai4bharat_org/EXhX84sbTQhLrsURCU9DlUwBVyJ10cYK9bQQe1SMljf_yA?download=1 -c -O 'v3.zip'
# !unzip v3.zip -d samanantar
# !rm -rf v3.zip


# %cd /content/samanantar/v2/en-te

# !rm -R -- */

# %cd /content/


# In[3]:


# # clone the repo for running evaluation
# !git clone https://github.com/AI4Bharat/indicTrans.git
# %cd indicTrans
# # clone requirements repositories
# !git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
# !git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
# !git clone https://github.com/rsennrich/subword-nmt.git
# %cd ..


# # Install the necessary libraries
# !pip install sacremoses pandas mock sacrebleu tensorboardX pyarrow indic-nlp-library
# ! pip install mosestokenizer subword-nmt
# # Install fairseq from source
# !git clone https://github.com/pytorch/fairseq.git
# %cd fairseq
# # !git checkout da9eaba12d82b9bfc1442f0e2c6fc1b895f4d35d
# !pip install ./
# ! pip install xformers
# %cd ..


# # add fairseq folder to python path
# import os
# os.environ['PYTHONPATH'] += ":/content/fairseq/"
# # sanity check to see if fairseq is installed
# from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils


# In[4]:


# # download the indictrans model

# %cd /content/
# # downloading the indic-en model
# !wget https://ai4b-my.sharepoint.com/:u:/g/personal/sumanthdoddapaneni_ai4bharat_org/ETnq-z4aHXFAjDF1Te3AZ20BaZ59PwlKlzSemEHhrmYJ3w?download=1 -c -O 'indic-en.zip'
# !unzip indic-en.zip
# !rm -rf indic-en.zip

# # downloading the en-indic model
# # !wget https://storage.googleapis.com/samanantar-public/V0.3/models/en-indic.zip
# # !unzip en-indic.zip

# # # downloading the indic-indic model
# # !wget https://storage.googleapis.com/samanantar-public/V0.3/models/m2m.zip
# # !unzip m2m.zip

# %cd /content/indicTrans


# In[5]:


# import os
# os.chdir("/content/indicTrans")
# from indicTrans.inference.engine import Model

# indic2en_model = Model(expdir='../indic-en')


# In[6]:


import datasets, evaluate
import random
import torch
import numpy as np

bertscore = evaluate.load("bertscore")
# comet = evaluate.load("comet", "wmt21-comet-qe-mqm", show_progress = True, cuda = torch.cuda.is_available())


# In[7]:


# comet


# In[8]:


bertscore


# In[9]:


def sample_sentences(lang, num = 100000):
    with open(f"/content/samanantar/v2/en-{lang}/train.en") as en_f, open(f"/content/samanantar/v2/en-{lang}/train.{lang}") as ind_f:
        sents = random.sample([(en, ind) for ind, en in zip(ind_f.readlines(), en_f.readlines())], num)
    
    outs = {
        "indic": [sent[1] for sent in sents],
        "english": [sent[0] for sent in sents]
    }
    return outs


# In[10]:


# from comet import download_model, load_from_checkpoint

# model_path = download_model("wmt21-comet-qe-mqm")
# comet = load_from_checkpoint(model_path)


# In[11]:


torch.cuda.empty_cache()
INDIC = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
datasets = [
    'indic-TOP',
             'itop',
             'indic-atis']


# In[12]:


from pprint import pprint


# def comet_score_data(sents):
#     data = {"src": sents["english"], "mt": sents['indic'], "ref": ["" for _ in range(len(sents['indic']))]}
#     data = [dict(zip(data, t)) for t in zip(*data.values())]
#     seg_scores, avg_score = comet.predict(data, batch_size=64, gpus=1)
#     return seg_scores, avg_score

def bertscore_data(sents):
    results = bertscore.compute(
                                predictions=sents['indic'], 
                                references=sents['english'], 
                                model_type="xlm-roberta-large", 
                                batch_size=128,
                                use_fast_tokenizer=True
                                )
    return results["f1"]


# In[13]:


from datasets import load_dataset, load_metric, Dataset, DatasetDict
import pandas as pd
import json

def create_dataset(dataset, train_lang, test_lang, backtranslated=False):

    if train_lang != "en":
        with open(f"Indic-SemParse/unfiltered_data/{dataset}/{train_lang}.json", "r") as f:
            train_data = json.load(f)

    else:
        with open(f"Indic-SemParse/unfiltered_data/{dataset}/hi.json", "r") as f:
            train_data = json.load(f)

    if test_lang != "en":
        with open(f"Indic-SemParse/unfiltered_data/{dataset}/{test_lang}.json", "r") as f:
            test_data = json.load(f)

    else:
        with open(f"Indic-SemParse/unfiltered_data/{dataset}/hi.json", "r") as f:
            test_data = json.load(f)

    train = train_data["train"]

    val = test_data["validation"]

    test = test_data["test"]

    if dataset == "itop":
        for idx, example in enumerate(train):
            if train_lang != "en":
                train[idx]["original_src"] = example["question"]
                train[idx]["translated_src"] = example["postprocessed_translated_question"]
            else:
                train[idx]["src"] = example["question"]
            train[idx]["trg"] = example["logical_form"]

        for idx, example in enumerate(val):
            if test_lang != "en":
                if backtranslated:
                    val[idx]["orig"] = example["question"]
                    val[idx]["src"] = example["backtranslated_post_processed_questions"]
                else:
                    val[idx]["orig"] = example["question"]
                    val[idx]["src"] = example["postprocessed_translated_question"]
            else:
                val[idx]["src"] = example["question"]
            val[idx]["trg"] = example["logical_form"]

        for idx, example in enumerate(test):
            if test_lang != "en":
                if backtranslated:
                    test[idx]["orig"] = example["question"]
                    test[idx]["src"] = example[
                        "backtranslated_post_processed_questions"
                    ]
                else:
                    test[idx]["orig"] = example["question"]
                    test[idx]["src"] = example["postprocessed_translated_question"]
            else:
                test[idx]["src"] = example["question"]
            test[idx]["trg"] = example["logical_form"]

    elif dataset == "indic-TOP":
        for idx, example in enumerate(train):
            if train_lang != "en":
                train[idx]["src"] = example["postprocessed_translated_question"]
            else:
                train[idx]["src"] = example["question"]
            train[idx]["trg"] = example["decoupled logical form"]

        for idx, example in enumerate(val):
            if test_lang != "en":
                if backtranslated:
                    val[idx]["orig"] = example["question"]
                    val[idx]["src"] = example["backtranslated_post_processed_questions"]
                else:
                    val[idx]["orig"] = example["question"]
                    val[idx]["src"] = example["postprocessed_translated_question"]
            else:
                val[idx]["src"] = example["question"]
            val[idx]["trg"] = example["decoupled logical form"]

        for idx, example in enumerate(test):
            if test_lang != "en":
                if backtranslated:
                    test[idx]["orig"] = example["question"]
                    test[idx]["src"] = example[
                        "backtranslated_post_processed_questions"
                    ]
                else:
                    test[idx]["orig"] = example["question"]
                    test[idx]["src"] = example["postprocessed_translated_question"]
            else:
                test[idx]["src"] = example["question"]
            test[idx]["trg"] = example["decoupled logical form"]
            

    elif dataset == "indic-atis":
        for idx, example in enumerate(train):
            if train_lang != "en":
                train[idx]["src"] = example["translated text"]
            else:
                train[idx]["src"] = example["text"]
            train[idx]["trg"] = example["logical form"]

        for idx, example in enumerate(val):
            if test_lang != "en":
                if backtranslated:
                    val[idx]["orig"] = example["text"]
                    val[idx]["src"] = example["back translated text"]
                else:
                    val[idx]["orig"] = example["text"]
                    val[idx]["src"] = example["translated text"]
            else:
                val[idx]["src"] = example["text"]
            val[idx]["trg"] = example["logical form"]

        for idx, example in enumerate(test):
            if test_lang != "en":
                if backtranslated:
                    test[idx]["orig"] = example["text"]
                    test[idx]["src"] = example["back translated text"]
                else:
                    test[idx]["orig"] = example["text"]
                    test[idx]["src"] = example["translated text"]
            else:
                test[idx]["src"] = example["text"]
            test[idx]["trg"] = example["logical form"]

    train_data = Dataset.from_pandas(pd.DataFrame(data=train))

    val_data = Dataset.from_pandas(pd.DataFrame(data=val))

    test_data = Dataset.from_pandas(pd.DataFrame(data=test))

    dataset = DatasetDict()

    dataset["train"] = train_data

    dataset["val"] = val_data

    dataset["test"] = test_data

    return dataset


# In[14]:


dataset = create_dataset("itop", "hi", "hi", True)

dataset['test']['src'][0], dataset['test']['orig'][0]


# In[15]:


from statistics import mean


# In[16]:


def bt_bertscore_data(sents):
    results = bertscore.compute(
                                predictions=sents['src'], 
                                references=sents['orig'], 
                                batch_size=256,
                                lang="en",
                                use_fast_tokenizer=True,
                                verbose=True,
                                idf=True
                                )
    return mean([round(v,2) for v in results["f1"]])


# In[ ]:


import json
import os
bt_bert_scores = []


print("calculating scores")
for i, dataset in enumerate(datasets):
    bt_bert_scores += [{"dataset":dataset}]
    for lang in INDIC:
        print(dataset, lang)
        bt_bert_scores[i][lang] = bt_bertscore_data(create_dataset(dataset, lang, lang, True)['test'])


pd.DataFrame(bt_bert_scores).to_csv("bt_bertscore_results.csv", index=False)


# In[ ]:





# In[ ]:


import os
import json
comet_scores = {}


for lang in INDIC:
    if "samantar_comet_scores.json" in os.listdir("/content/drive/MyDrive/iTOP/"):
        with open("/content/drive/MyDrive/iTOP/samantar_comet_scores.json", "r") as f:
            comet_scores = json.load(f)
        if lang in comet_scores:
            continue
    print(lang)
    sents = sample_sentences(lang)
    comet_scores[lang] = round(comet_score_data(sents)[1], 3)
    pprint(comet_scores)
    torch.cuda.empty_cache()
    with open("/content/drive/MyDrive/iTOP/samantar_comet_scores.json", "w") as f:
        json.dump(comet_scores, f, indent = 6)



# In[ ]:


import json
bert_scores = {}

offset = 10000

for lang in INDIC:
    if "samantar_bert_scores.json" in os.listdir("/content/drive/MyDrive/iTOP/"):
        with open("/content/drive/MyDrive/iTOP/samantar_bert_scores.json", "r") as f:
            bert_scores = json.load(f)
        if lang in bert_scores:
            continue
    print(lang)
    scores = []
    sents = sample_sentences(lang)
    for idx in range(0, len(sents), offset):
        scores += bertscore_data({k:sents[k][idx:idx+offset] for k in sents})
    bert_scores[lang] = sum(scores)/len(scores)
    pprint(bert_scores)
    torch.cuda.empty_cache()
  
    with open("/content/drive/MyDrive/iTOP/samantar_bert_scores.json", "w") as f:
        json.dump(bert_scores, f, indent = 6)


# In[ ]:


try:
    os.mkdir("/content/drive/MyDrive/iTOP/filtered_data")
except:
    pass


# In[ ]:


try:
    os.mkdir("/content/drive/MyDrive/iTOP/filtered_data/itop")
except:
    pass

try:
    os.mkdir("/content/drive/MyDrive/iTOP/filtered_data/indic-TOP")
except:
    pass

try:
    os.mkdir("/content/drive/MyDrive/iTOP/filtered_data/indic-atis")
except:
    pass


# In[ ]:


import copy


# In[ ]:


try:
    os.mkdir("/content/drive/MyDrive/iTOP/filtered_data_2")
except:
    pass

try:
    os.mkdir("/content/drive/MyDrive/iTOP/filtered_data_2/itop")
except:
    pass

try:
    os.mkdir("/content/drive/MyDrive/iTOP/filtered_data_2/indic-TOP")
except:
    pass

try:
    os.mkdir("/content/drive/MyDrive/iTOP/filtered_data_2/indic-atis")
except:
    pass


# In[ ]:


with open("/content/drive/MyDrive/iTOP/samantar_comet_scores.json", "r") as f:
    comet_scores = json.load(f)

with open("/content/drive/MyDrive/iTOP/samantar_bert_scores.json", "r") as f:
    bert_scores = json.load(f)


# In[ ]:


for ds in data_sets:
    for lang in INDIC:
        print(ds)
        print(lang)
        if f"{lang}.json" in os.listdir(f"/content/drive/MyDrive/iTOP/filtered_data_2/{ds}"):
            print("skipping.....")
            continue

        data  = create_dataset(ds,lang, lang)

        print("val")
        dev_comet_scores = comet_score_data({"english": data['val']['orig'], "indic": data['val']['src']})[0]
        dev_bert_scores = bertscore_data({"english": data['val']['orig'], "indic": data['val']['src']})

        print("test")
        test_comet_scores = comet_score_data({"english": data['test']['orig'], "indic": data['test']['src']})[0]
        test_bert_scores = bertscore_data({"english": data['test']['orig'], "indic": data['test']['src']})
        

        print("test comet Score:", np.mean(test_comet_scores))
        print("test bert Score:", np.mean(test_bert_scores))
        

        new_data = DatasetDict()

        new_data['train'] = data['train']
        new_data['val'] = data['val'].select(indices=[idx for idx in range(len(data['val'])) 
                                                        if dev_comet_scores[idx] >= 0.9*comet_scores[lang] and 
                                                            dev_bert_scores[idx] >= 0.9*bert_scores[lang]])
        
        new_data['test'] = data['test'].select(indices=[idx for idx in range(len(data['test'])) 
                                                        if test_comet_scores[idx] >= 0.9*comet_scores[lang] and 
                                                            test_bert_scores[idx] >= 0.9*bert_scores[lang]])
        

        print(new_data)
        data_save = {k:new_data[k].to_pandas() for k in new_data.keys()}
        if ds == "indic-atis":
            
            data_save['train']['entities'] = list(map(json.dumps, list(map(list,new_data['train'].to_pandas()['entities'].to_list()))))
            data_save['val']['entities'] = list(map(json.dumps, list(map(list,new_data['val'].to_pandas()['entities'].to_list()))))
            data_save['test']['entities'] = list(map(json.dumps, list(map(list,new_data['test'].to_pandas()['entities'].to_list()))))

        with open(f"/content/drive/MyDrive/iTOP/filtered_data/{ds}/{lang}.json", "w", encoding="utf-8") as f:
            json.dump({k:data_save[k].to_dict("records") for k in data_save}, f, indent=6, ensure_ascii=False)

        # else:
        #     with open(f"/content/drive/MyDrive/iTOP/filtered_data/{ds}/{lang}.json", "w", encoding="utf-8") as f:
        #         json.dump({k:new_data[k].to_pandas().to_dict("records") for k in data_save}, f, indent=6, ensure_ascii=False)




# In[ ]:




