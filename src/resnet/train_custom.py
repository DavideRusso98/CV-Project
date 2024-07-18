import argparse
import os
from datetime import time

import torch
import torchvision
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN

from resnet.components import KeypointHead, AutomotiveKeypointDetector
from resnet.utils import COCODataset, get_transform, MetricLogger, SmoothedValue
from test_resnet.coco_eval import CocoEvaluator


def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = data_loader.dataset.coco
    iou_types = ["bbox", "keypoints"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator


def compute_loss(loss_dict, alpha=1.):
    loss_classifier = loss_dict['loss_classifier']
    loss_box_reg = loss_dict['loss_box_reg']
    loss_keypoint = loss_dict['loss_keypoint']
    loss_objectness = loss_dict['loss_objectness']
    loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']
    return sum([loss_classifier, loss_box_reg, alpha * loss_keypoint, loss_objectness, loss_rpn_box_reg])


def train_epoch(data_loader, model, optimizer, device, epoch):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    for images, targets in metric_logger.log_every(data_loader, 20, f"Epoch: [{epoch}]"):
        optimizer.zero_grad()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                   for t in targets]
        loss_dict = model(images, targets)
        losses = compute_loss(loss_dict, alpha=1.1)
        losses.backward()
        optimizer.step()
        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        
def modify_resnet_dilation(backbone, dilation=2):
    for name, module in backbone.named_children():
        if 'layer4' in name:  # Modifica solo layer4 per questo esempio
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Conv2d):
                    sub_module.dilation = (dilation, dilation)
                    sub_module.padding = (dilation, dilation)
                elif isinstance(sub_module, nn.Sequential):
                    for nn_name, nn_module in sub_module.named_children():
                        if isinstance(nn_module, nn.Conv2d):
                            nn_module.dilation = (dilation, dilation)
                            nn_module.padding = (dilation, dilation)
    return backbone


def get_model(num_classes=2, num_keypoints=20, trainable_layers=3):
    backbone = resnet50(progress=True, norm_layer=nn.BatchNorm2d)
    backbone = modify_resnet_dilation(backbone)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers)
    keypoint_head = KeypointHead(backbone.out_channels, tuple(512 for _ in range(10)))
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                       aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, num_classes)
    return KeypointRCNN(backbone,
                         num_keypoints=num_keypoints,
                         keypoint_head=keypoint_head,
                         anchor_generator=anchor_generator,
                         box_predictor=box_predictor)


def main():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('images', type=str, help='Path to images dir')
    parser.add_argument('train', type=str, help='Path to train .json')
    parser.add_argument('--test', type=str, help='Path to test .json')
    parser.add_argument('--out', '-o', required=True, dest='output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    dataset = COCODataset(args.images, args.train, get_transform())
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)))

    NUM_EPOCHS = 8
    model = AutomotiveKeypointDetector()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f"Using device: {device}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_epoch(data_loader, model, optimizer, device, epoch)
        lr_scheduler.step()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'custom_dilation.pth'))
    print('model saved')


if __name__ == '__main__':
    main()
