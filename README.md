# CV-Project
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wDojrxs4_KjhYU6Cr-fyJ-_waBEcbizh?usp=sharing)

## Link to our Google Drive
[DATASET KEYPOINT](https://drive.google.com/drive/folders/1GCpRsDSXSHfqCM5T36EM5d35a_DVg3LB?usp=drive_link)


## Environment config

### Venv
How to activate venv
Alessio:
```bash
source /home/alesvale/.virtualenvs/CV-Project/bin/activate
```

### Conda
Create Conda Environment with python version 3.x
```bash
conda create --name env_339887 -f environment.yml python=3.x
```

Export all conda's dependencies to an environment.yml file:
```bash
conda env export --from-history > environment.yml
```

## Dataset creation

### Launch main via blenderproc. Before launching make sure to cd into project root 
```bash
blenderproc run ./src/dataset/main.py ./src/resources/objects3d/ ./src/dataset/output/
```

### Visualize image with its keypoints in 2D
```bash
#usage: visualize.py [-h] [model_name] [output_dir]
#positional arguments:
#  model_name  car model name
#  output_dir  Path to output/ dir

python src/dataset/visualize.py tesla_0 ./src/dataset/output
```

## Training

### Split dataset into train and test 80/20
```bash
 python ./src/dataset/cocosplit.py --having-annotations -s 0.9 ./src/dataset/output/coco_annotations.json ./src/dataset/output/coco_train_1800.json ./src/dataset/output/coco_test_200.json
```


WS di Riccardo:
```bash
python ./src/resnet/inference.py /home/riccardo/Scrivania/UNIVERSITA/Computer Vision and Cognitive Systems/CV-Project/src/resnet/automotive_keypoint_detector.pth ./src/resnet/inference.py /home/riccardo/Scrivania/UNIVERSITA/Computer Vision and Cognitive Systems/CV-Project/src//dataset/output/images/clean/alfa_0.jpg
```

Alessio:
```bash
python ./src/resnet/inference.py ./src/resnet/akd_11.pth ./src/dataset/output/images/clean/alfa_0.jpg
```