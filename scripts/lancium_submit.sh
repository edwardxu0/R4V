#!/usr/bin/env bash

# ./scripts/lancium_package.sh

configdir=${1%/}; shift

count=0
for config in $configdir/*; do
    count=$(($count+1))

    uuid="$(python -c "import uuid; print(str(uuid.uuid4()).lower())")"
    echo "./scripts/lancium_submit.py"\
        "-G rns:/home/CCC/Lancium/dls2fc@virginia.edu/foo"\
        "-I r4v.tar.gz"\
        "-O r4v/tmp"\
        "--gres=gpu:1" \
        "--error \"$uuid.err\""\
        "--output \"$uuid.out\""\
        "/usr/local/bin/singularity-wrapper pytorch"\
        "bash ./r4v/scripts/lancium_run.sh \"$config\" \"$uuid\""
    ./scripts/lancium_submit.py \
        -G rns:/home/CCC/Lancium/dls2fc@virginia.edu/foo \
        -I r4v.tar.gz \
        -O r4v/tmp \
        --gres=gpu:1 \
        --error "$uuid.err" \
        --output "$uuid.out" \
        /usr/local/bin/singularity-wrapper pytorch \
        bash ./r4v/scripts/lancium_run.sh "$config" "$uuid"
    if [ $count -eq 2 ]
    then
        break
    fi
done
