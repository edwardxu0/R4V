#!/usr/bin/env bash

# print out some debug info
date
echo $(hostname)
echo "[PATH]/lancium_run.sh $@"

# set up environment
# export PYTHONPATH=$(pwd)/.venv/lib/python3.6/site-packages:$PYTHONPATH
# pip install --user toml
# pip install --user onnx
conda init bash
conda create -y -p $(pwd)/.venv python=3.6
conda activate $(pwd)/.venv
conda install -y pytorch=1.0.0 torchvision=0.2.1 cudatoolkit=10.0 -c pytorch
conda install -y numpy=1.15.4 toml=0.10.0
pip install --user onnx

cd r4v

# prepare the config file
filename=$(basename $1)
identifier=$2
echo "$identifier"

filename=$(basename $1)
config_name="${filename%.*}"
echo $config_name
mkdir -p tmp/$config_name/$identifier
config=tmp/$config_name/$identifier/config.toml
echo $config
cat $1 > $config
echo "[distillation.student]" >> $config
echo "path=\"tmp/$config_name/$identifier/model.onnx\"" >> $config

echo

# run distillation
echo "python -m r4v distill $config -v"
python -m r4v distill $config -v