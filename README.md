# CV-Project
## Introduction
We present a novel framework for automotive keypoint detection. Our approach tackles the significant challenges posed by the variability in shapes, sizes, and appearances of vehicles. Traditional methods relying on manual annotation are both time-consuming and error-prone, while models trained on synthetic data often suffer from domain-shift when applied to real-world images. To overcome these issues, we developed a comprehensive keypoint detection system inspired by FastTrakAI, leveraging a robust dataset of over 2000 images annotated with keypoints.
Our methodology includes the creation of a synthetic dataset using 3D models of various vehicles, automated keypoint annotation, and training of a keypoint detection model tailored to these annotations. We employ a ResNet-50 backbone with a modified Feature Pyramid Network and an enhanced keypoint head incorporating residual blocks for improved keypoint prediction accuracy. The model's performance is evaluated on both synthetic images and real-world captures, ensuring robustness and generalization to unseen data.
Furthermore, we introduce an advanced data retrieval system utilizing Object Keypoint Similarity (OKS) metrics to identify the most similar keypoints across different images, enhancing the alignment of synthetic and real image domains. This system, combined with an offline database approach for precomputed keypoint scores, significantly improves retrieval speed and accuracy.

## Link to our Google Drive
[ANNOTATED DATASET](https://drive.google.com/drive/folders/1GCpRsDSXSHfqCM5T36EM5d35a_DVg3LB?usp=drive_link)


## Detection Result
![Detections](./img/final_annotated.png)

## Retrieval output
![Retrieval](./img/final_annotated.png)

