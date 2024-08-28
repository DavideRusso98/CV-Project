# CV-Project
## Introduction
We present a novel framework for automotive keypoint detection. Our approach tackles the significant challenges posed by the variability in shapes, sizes, and appearances of vehicles. Traditional methods relying on manual annotation are both time-consuming and error-prone, while models trained on synthetic data often suffer from domain-shift when applied to real-world images. To overcome these issues, we developed a comprehensSketchfabive keypoint detection system inspired by FastTrakAI, leveraging a robust dataset of over 2000 images annotated with keypoints.
Our methodology includes the creation of a synthetic dataset using 3D models of various vehicles, automated keypoint annotation, and training of a keypoint detection model tailored to these annotations. We employ a ResNet-50 backbone with a modified Feature Pyramid Network and an enhanced keypoint head incorporating residual blocks for improved keypoint prediction accuracy. The model's performance is evaluated on both synthetic images and real-world captures, ensuring robustness and generalization to unseen data.
Furthermore, we introduce an advanced data retrieval system utilizing Object Keypoint Similarity (OKS) metrics to identify the most similar keypoints across different images, enhancing the alignment of synthetic and real image domains. This system, combined with an offline database approach for precomputed keypoint scores, significantly improves retrieval speed and accuracy.

## Link to our Google Drive
[ANNOTATED DATASET](https://drive.google.com/drive/folders/1GCpRsDSXSHfqCM5T36EM5d35a_DVg3LB?usp=drive_link)


## Dataset creation

### Launch main via blenderproc. Before launching make sure to cd into project root 
```bash
blenderproc run ./src/dataset/main.py ./src/resources/3d_models/test ./src/dataset/output2 0 --init-coco --camera_poses 10
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
