# CV-Project
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wDojrxs4_KjhYU6Cr-fyJ-_waBEcbizh?usp=sharing)


Some commands

### activate alesvale's venv
```bash
source /home/alesv/.virtualenvs/CV-Project/bin/activate
```

### Launch main via blenderproc. Before launching make sure to cd into project root 
```bash
blenderproc run ./src/dataset/main.py ./src/resources/objects3d/tesla.blend ./src/dataset/output/
```

### Launch example1 via blenderproc. Before launching make sure to cd into project root
```bash
blenderproc run ./src/dataset/example1.py ./src/resources/camera_positions ./src/resources/objects3d/tesla.blend ./src/dataset/output/
```

### Visualize image with its keypoints in 2D
```bash
#usage: visualize.py [-h] [model_name] [output_dir]
#positional arguments:
#  model_name  car model name
#  output_dir  Path to output/ dir

python src/dataset/visualize.py tesla_0 ./src/dataset/output
```
