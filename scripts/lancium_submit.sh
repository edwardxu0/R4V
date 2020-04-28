#!/usr/bin/env bash

if [[ "$1" == "setup" ]]; then
    echo "./scripts/lancium_submit.py" \
        "-J \"lanciumSetup\"" \
        "-G rns:/home/CCC/Lancium/dls2fc@virginia.edu/foo" \
        "-I r4v.tar.gz" \
        "--gres=gpu:1" \
        "--error \"setup.err\"" \
        "--output \"setup.out\"" \
        "--time 12:00:00" \
        "--mem 16G" \
        "/nfs/software/wrappers/py36-gcc-wrapper" \
        "./r4v/scripts/lancium_setup.sh"
    ./scripts/lancium_submit.py \
        -J "lanciumSetup" \
        -G rns:/home/CCC/Lancium/dls2fc@virginia.edu/foo \
        -I r4v.tar.gz \
        --gres=gpu:1 \
        --error "setup.err" \
        --output "setup.out" \
        --time 12:00:00 \
        --mem 16G \
        /nfs/software/wrappers/py36-gcc-wrapper \
        ./r4v/scripts/lancium_setup.sh
elif [[ "$1" == "distill" ]]; then
    shift
    configdir=${1%/}
    shift

    for i in 1 2 3 4 5; do
        for config in $configdir/*; do
            count=$(($count + 1))

            uuid="$(python -c "import uuid; print(str(uuid.uuid4()).lower())")"
            echo "./scripts/lancium_submit.py" \
                "-J \"distill-$uuid\"" \
                "-G rns:/home/CCC/Lancium/dls2fc@virginia.edu/foo" \
                "-I r4v.tar.gz" \
                "-O $uuid.model.tar.gz" \
                "--gres=gpu:1" \
                "--error \"$uuid.err\"" \
                "--output \"$uuid.out\"" \
                "--time 12:00:00" \
                "--mem 8G" \
                "/nfs/software/wrappers/py36-gcc-wrapper" \
                "./r4v/scripts/lancium_run.sh \"$config\" \"$uuid\""
            ./scripts/lancium_submit.py \
                -J "distill-$uuid" \
                -G rns:/home/CCC/Lancium/dls2fc@virginia.edu/foo \
                -I r4v.tar.gz \
                -O $uuid.model.tar.gz \
                --gres=gpu:1 \
                --error "$uuid.err" \
                --output "$uuid.out" \
                --time 12:00:00 \
                --mem 8G \
                /nfs/software/wrappers/py36-gcc-wrapper \
                ./r4v/scripts/lancium_run.sh "$config" "$uuid"
        done
    done
fi
