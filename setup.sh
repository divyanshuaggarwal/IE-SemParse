#bin/bash

pip install -r requirements.txt

echo "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" > ~/.huggingface

apt-get update && apt-get upgrade -y

apt-get install tmux git-lfs htop

git lfs install
git clone https://divyanshu:Wateringplants_98@huggingface.co/datasets/Divyanshu/Indic-SemParse

tmux

python autostop.py

tmux detach
