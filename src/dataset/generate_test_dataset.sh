ROOT_PATH="/mnt/c/Users/alesv/PycharmProjects/AutomotiveKeypointDetector/"
RELATIVE_OUTPUT_PATH="./src/dataset/output_test"
cd $ROOT_PATH || exit

echo "Generating dataset";

blenderproc run ./src/dataset/main.py ./src/resources/3d_models/test "$RELATIVE_OUTPUT_PATH" 0 --init-coco
