import argparse
from datetime import datetime, time

import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
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


def train_epoch(data_loader, model, optimizer, device, epoch):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    for images, targets in metric_logger.log_every(data_loader, 20, f"Epoch: [{epoch}]"):
        optimizer.zero_grad()

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                   for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def main():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('images', type=str, help='Path to images dir')
    parser.add_argument('train', type=str, help='Path to train .json')
    parser.add_argument('--test', type=str, help='Path to test .json')
    args = parser.parse_args()

    dataset = COCODataset(args.images, args.train, get_transform())

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)))

    num_keypoints = 20
    num_classes = 2
    num_epochs = 8

    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = keypointrcnn_resnet50_fpn(num_classes=num_classes,
                                      num_keypoints=num_keypoints,
                                      anchor_generator=anchor_generator)
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features, num_keypoints)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features,
        num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f"Using device: {device}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        train_epoch(data_loader, model, optimizer, device, epoch)
        lr_scheduler.step()

    torch.save(model.state_dict(), f'./src/resnet/trained_models/akd-1.0.pth')
    print('model saved')
    print(f'epochs: {num_epochs}')


if __name__ == '__main__':
    main()
