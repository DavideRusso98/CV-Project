import argparse
import os

import time
import torch
import torchvision
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor, KeypointRCNN

from resnet.components import AutomotiveKeypointDetector, KeypointHead
from resnet.train_default import COCODataset
from resnet.utils import get_transform, plot_keypoints, threshold_keypoints, MetricLogger, compute_coco_areas, \
    extract_probs_from_scores
from test_resnet.coco_eval import CocoEvaluator


def get_default_model(num_classes=2, num_keypoints=20):
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                       aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = keypointrcnn_resnet50_fpn(num_classes=num_classes,
                                      num_keypoints=num_keypoints,
                                      anchor_generator=anchor_generator)
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features, num_keypoints)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return model

### Model linked to akd-2.0.pth
def get_custom_model(num_classes=2, num_keypoints=20, trainable_layers=3):
    backbone = resnet50(progress=True, norm_layer=nn.BatchNorm2d)
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
    parser = argparse.ArgumentParser(description='Inference keypoints')
    parser.add_argument('model', type=str, help='Path to .pth model file.')
    parser.add_argument('--out', '-o', dest='dest_dir', type=str, help='Path to output_image')
    parser.add_argument('--images', dest='image_folder', type=str, help='Path to image folder')
    parser.add_argument('--coco', '-c', dest='coco_test', type=str, help='Path to .json coco annotations file')

    args = parser.parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)

    device = torch.device('cuda')
    #model = AutomotiveKeypointDetector()
    #model = get_default_model()
    model = get_custom_model()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    dataset_test = COCODataset(args.image_folder, args.coco_test, get_transform())
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)))

    coco = data_loader_test.dataset.coco
    coco = compute_coco_areas(coco)
    iou_types = ["bbox", "keypoints"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    metric_logger = MetricLogger(delimiter="  ")

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader_test, 10, "Test:"):

            model_time = time.time()
            predictions = model(images)
            model_time = time.time() - model_time
            for i, image in enumerate(images):
                image = image.cpu().numpy().transpose((1, 2, 0))
                boxes = predictions[i]['boxes'].cpu().detach().numpy()
                keypoints = predictions[i]['keypoints'].cpu().detach().numpy()
                scores = predictions[i]['scores'].cpu().detach().numpy()
                max_score_index = scores.argmax()
                actual_keypoints = targets[i]['keypoints'][0]
                pred_keypoints = keypoints[max_score_index]
                keypoints_scores = predictions[i]['keypoints_scores'][max_score_index]
                keypoints_prob = extract_probs_from_scores(keypoints_scores)
                actual_bbox = targets[i]['boxes'][0]
                pred_bbox = torch.tensor(boxes[max_score_index]).reshape(4)
                thresh_keypoint = threshold_keypoints(pred_keypoints, keypoints_prob, 0.3)
                plot_keypoints(image, actual_keypoints, thresh_keypoint, actual_bbox, pred_bbox, args.dest_dir, targets[i]['img_name'])

                ### Evaluation Metrics
                predictions = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in predictions]
                res = {target["image_id"]: output for target, output in zip(targets, predictions)}
                evaluator_time = time.time()
                coco_evaluator.update(res)
                evaluator_time = time.time() - evaluator_time
                metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        coco_evaluator.accumulate()
        coco_evaluator.summarize()

if __name__ == '__main__':
    main()
