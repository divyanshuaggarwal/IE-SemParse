#bin/bash

# pip install pipreqs
# pipreqs . --force

pip install -r requirements.txt

echo "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" > ~/.huggingface

apt-get update && apt-get upgrade -y

apt-get install tmux git-lfs htop -y

git lfs install
git clone https://divyanshu:Wateringplants_98@huggingface.co/datasets/Divyanshu/Indic-SemParse 
# tmux

# python autostop.py

# tmux detach

git config --global user.email "divyanshuggrwl@gmail.com"
git config --global user.name "divyanshuaggarwal"


mkdir ~/.cache/huggingface/
mkdir ~/.cache/huggingface/accelerate/

cp accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

mkdir results/
mkdir models/