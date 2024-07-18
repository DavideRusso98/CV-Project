
Experiments

| Model name | Description                    | Backbone type | # Epochs | KeypointHeadDepth | alpha | dilation | bbox AP | bbox AR | keypoint AP | keypoint AR |
|------------|--------------------------------|---------------|:--------:|:-----------------:|:-----:|:--------:|:-------:|:-------:|:-----------:|:-----------:|
| akd-1.0    | torchvision default model      | resnet50      |    8     |       none        |  1.1  |    1     |   0.9   |   0.9   |     0.6     |     0.6     | 
| akd-2.0    | Custom KeypointHead            | resnet50      |    8     |        10         |  1.1  |    1     |   0.9   |   0.9   |     0.1     |     0.1     |
| akd-2.1    | Custom KH and changed dilation | resnet50      |    8     |         4         |  1.2  |    2     |  0.925  |  0.950  |     0.7     |     0.7     |

