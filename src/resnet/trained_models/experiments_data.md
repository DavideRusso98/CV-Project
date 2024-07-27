
Experiments

| Model name | Description                         | Backbone type      | # Epochs | KeypointHeadDepth | alpha | dilation | bbox AP | bbox AR | keypoint AP | keypoint AR |
|------------|-------------------------------------|--------------------|:--------:|:-----------------:|:-----:|:--------:|:-------:|:-------:|:-----------:|:-----------:|
| akd-1.0    | torchvision default model           | resnet50           |    8     |       none        |  1.1  |    1     |  0.900  |  0.900  |    0.600    |    0.600    | 
| akd-2.0    | Custom KeypointHead                 | resnet50           |    8     |        10         |  1.1  |    1     |  0.900  |  0.900  |    0.105    |    0.100    |
| akd-2.1    | Custom KH and changed dilation      | resnet50           |    8     |         4         |  1.2  |    2     |  0.925  |  0.950  |    0.700    |    0.700    |
| akd-2.2    | Same as above                       | resnet50           |    11    |         2         |   1   |    2     |  0.900  |  0.900  |    0.600    |    0.600    |
| akd-2.3    | Same as above                       | resnet50           |    11    |         6         |  1.2  |    2     |  0.900  |  0.900  |    0.800    |    0.800    |
| akd-2.4    | Same as above                       | resnet50           |    15    |         6         |  1.2  |    2     |  1.000  |  1.000  |    0.700    |    0.700    |
| akd-3.0    | Custom KH, custom dilation, diff BB | resnext50_32x4d    |    8     |         4         |  1.2  |    2     |  0.850  |  0.900  |    0.350    |    0.400    |

