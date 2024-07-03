# from resnet.model import resnet_test

# if __name__ == '__main__':
#     resnet_test()

from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.optim as optim
import torchvision
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
transform = T.Compose([T.ToTensor()])

class CocoKeypoints(CocoDetection):
    def __getitem__(self, idx):
        img, target = super(CocoKeypoints, self).__getitem__(idx)
        img_id = self.ids[idx]
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        
        # Converti le annotazioni COCO in un formato adatto
        keypoints = []
        for obj in anno:
            keypoints.append(obj['keypoints'])
        
        target = {
            'image_id': img_id,
            'annotations': anno
        }
        return img, target
# Check if GPU's available, otherwise CPU is used.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', device)
# Percorsi ai file di annotazioni e immagini
ann_file = './src/dataset/output/coco_annotations.json'
img_dir = './src/dataset/output/images/'

# Caricamento delle annotazioni COCO
coco = COCO(ann_file)

# Creazione di un dataset COCO
dataset = datasets.CocoDetection(img_dir, ann_file, transform=transforms.ToTensor())
num_classes = len(coco.getCatIds()) + 1  # numero di categorie più 1 per lo sfondo
model = keypointrcnn_resnet50_fpn(pretrained=False,
                                  pretrained_backbone=True,
                                  num_classes=num_classes)

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features,
    num_classes
)
# print(num_classes)


# Parametri dell'ottimizzatore
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Creazione del DataLoader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Spostare il modello sul dispositivo
model.to(device)
# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        
        images = list(image.to(device) for image in images)
        dict_list = targets[0]

        # targets = [{k: (torch.tensor(v) if k == "boxes" else v) for k, v in t.items()} for t in dict_list]
        # targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in dict_list]
      

        corrected_targets = []
        for t in dict_list:
            corrected_target = {}
            for k, v in t.items():
                if k == "boxes":
                    corrected_target[k] = torch.tensor(v).to(device)
                    # print(type(v)," boxe")
                elif k == "labels":
                    corrected_target[k] = torch.tensor(v).to(device)
                elif k == "keypoints":
                    corrected_target[k] = torch.tensor(v).to(device)
            corrected_targets.append(corrected_target)
        targets = corrected_targets
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch: {epoch}, Loss: {losses.item()}")

# Validation
model.eval()
with torch.no_grad():
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        dict_list = targets[0]
        corrected_targets = []
        for t in dict_list:
            corrected_target = {}
            for k, v in t.items():
                if k == "boxes":
                    corrected_target[k] = torch.tensor(v).to(device)
                    # print(type(v)," boxe")
                elif k == "labels":
                    corrected_target[k] = torch.tensor(v).to(device)
                elif k == "keypoints":
                    corrected_target[k] = torch.tensor(v).to(device)
            corrected_targets.append(corrected_target)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in dict_list]
        targets = corrected_targets
        outputs = model(images)

# Saving model
torch.save(model.state_dict(), 'resnet_coco.pth')