
### activate alesvale's venv
```bash
source /home/alesv/.virtualenvs/CV-Project/bin/activate
```

### Launch main via blenderproc. Before launching make sure to cd into project root 
```bash
blenderproc run ./src/dataset/main.py ./src/resources/objects3d/tesla.blend ./src/dataset/output/keypoints2d.json
```