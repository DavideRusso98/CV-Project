import torch
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T


class COCODataset(CocoDetection):
    def __getitem__(self, idx):
        img, target = super(COCODataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        bbox_cxcyhw = torch.tensor(anno[0]['boxes']).reshape(1, 4)
        target = {
            'boxes': torchvision.ops.box_convert(bbox_cxcyhw, 'cxcywh', 'xyxy'),
            'labels': torch.tensor([1 for _ in range(1)], dtype=torch.int64),
            'keypoints': torch.tensor(anno[0]['keypoints']).reshape(1, -1, 3),
        }
        return img, target


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return torchvision.transforms.Compose(transforms)


# Carica il dataset
images_path = '/mnt/c/Users/alesv/PycharmProjects/CV-Project/src/dataset/output/images/clean/'
train_annotation_path = '/mnt/c/Users/alesv/PycharmProjects/CV-Project/src/dataset/output/coco_train.json'
test_annotation_path = '/mnt/c/Users/alesv/PycharmProjects/CV-Project/src/dataset/output/coco_test.json'

dataset = COCODataset(images_path, train_annotation_path, get_transform())
dataset_test = COCODataset(images_path, test_annotation_path, get_transform())

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)))

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)))

model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 2
num_keypoints = 20

in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
model.roi_heads.keypoint_predictor = torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(in_features,
                                                                                                      num_keypoints)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 3

model.train()
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        for key, value in loss_dict.items():
            print(f"{key}: {value}", end=' ')
        print('')
        loss = loss_dict['loss_keypoint']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr_scheduler.step()

torch.save(model.state_dict(), 'our_super_model_trained.pth')
