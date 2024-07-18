#!/usr/bin/env bash

ROOT_PATH="/mnt/c/Users/alesv/PycharmProjects/AutomotiveKeypointDetector/"
RELATIVE_OUTPUT_PATH="./src/dataset/output2"
cd $ROOT_PATH || exit

echo "Generating dataset";

blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 0 --init-coco &&
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 1 &&
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 2 &&
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 3 &&
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 4 &&
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 5 &&
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 6 &&
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 7 &&
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 8 &&
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/train "$RELATIVE_OUTPUT_PATH" 9;
