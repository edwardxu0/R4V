#!/bin/bash
set -e

virtualenv -p python3.6 .venv
. .venv/bin/activate

while read req || [ -n "$req" ]
do
    echo "pip install $req"
    pip install $req
done < requirements.txt
