
Metriche senza Custom KeypointHead

| Model name | Backbone type |  # Epochs  | alpha | bbox AP | bbox AR | keypoint AP | keypoint AR |
|------------|---------------|:----------:|:-----:|:-------:|:-------:|:-----------:|:-----------:|
| akd-1.0    | resnet50      |     8      |  1.1  |   1.0   |   1.0   |     1.0     |     1.0     |  


Metriche con Custom KeypointHead:

| Model name | Backbone type |  # Epochs  | KeypointHeadDepth | alpha | bbox AP | bbox AR | keypoint AP | keypoint AR |
|------------|---------------|:----------:|:-----------------:|:-----:|:-------:|:-------:|:-----------:|:-----------:|
| akd-2.0    | resnet50      |     8      |        10         |  1.1  |  0.950  |  0.950  |    0.802    |    0.800    |  

