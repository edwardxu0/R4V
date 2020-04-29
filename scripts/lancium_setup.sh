#!/usr/bin/env bash

# print out some debug info
date
echo $(hostname)
echo "$0 $@"

echo
echo "$ ls scratch"
ls scratch
echo

if [ "$1" == "rebuild" ]; then
    echo "Re-building environment"
    rm -rf ./scratch/.venv
    rm -rf ./scratch/DNNV
fi

# set up environment (do once)
if [ ! -e ./scratch/.venv/bin/activate ]; then
    cd scratch
    echo "Setting up execution environment..."
    echo $(python -V)
    echo $(which python)
    python -m venv .venv
    . .venv/bin/activate
    echo $(python -V)
    echo $(which python)

    python -m pip install --upgrade pip setuptools flit

    git clone https://github.com/dlshriver/DNNV.git
    cd DNNV
    git checkout develop
    flit install -s
    cd ..

    while read req || [ -n "$req" ]; do
        echo "pip install $req"
        pip install $req
    done <../r4v/requirements.txt
    deactivate
    cd ..
fi
