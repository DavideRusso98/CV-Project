import argparse
import os

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from resnet.train import COCODataset
from resnet.utils import get_transform, plot_keypoints

semantic_keypoints = [
    "wheel_FL",
    "wheel_FR",
    "wheel_RL",
    "wheel_RR",
    "door_FL",
    "door_FR",
    "door_RL",
    "door_RR",
    "headlight_FL",
    "headlight_FR",
    "headlight_RL",
    "headlight_RR",
    "windshield_FRONT",
    "windshield_REAR",
    "bumper_FRONT",
    "bumper_REAR",
    "license_plate",
    "mirror_LEFT",
    "mirror_RIGHT",
    "roof",
]


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),  # Converti l'immagine in tensore
    ])
    image = transform(image)
    return image


def log_predictions(predictions):
    scores = predictions[0]['keypoints_scores'].cpu()
    # Calcola statistiche descrittive
    all_scores = scores[0].flatten().numpy()
    mean_score = all_scores.mean()
    median_score = np.median(all_scores)
    min_score = all_scores.min()
    max_score = all_scores.max()
    print(f"Mean Score: {mean_score}")
    print(f"Median Score: {median_score}")
    print(f"Min Score: {min_score}")
    print(f"Max Score: {max_score}")


def main():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('model', type=str, help='Path to .pth model file.')
    parser.add_argument('--out', '-o', dest='dest_dir', type=str, help='Path to output_image')
    parser.add_argument('--images', dest='image_folder', type=str, help='Path to image folder')
    parser.add_argument('--coco', '-c', dest='coco_test', type=str, help='Path to .json coco annotations file')

    args = parser.parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)

    device = torch.device('cuda')

    model = keypointrcnn_resnet50_fpn(num_keypoints=20)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    dataset_test = COCODataset(args.image_folder, args.coco_test, get_transform())
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)))

    with torch.no_grad():
        for images, targets in data_loader_test:
            predictions = model(images)
            for i, image in enumerate(images):
                image = image.cpu().numpy().transpose((1, 2, 0))
                boxes = predictions[i]['boxes'].cpu().detach().numpy()
                keypoints = predictions[i]['keypoints'].cpu().detach().numpy()
                scores = predictions[i]['scores'].cpu().detach().numpy()
                max_score_index = scores.argmax()
                actual_keypoints = targets[i]['keypoints'][0]
                pred_keypoints = keypoints[max_score_index]
                actual_bbox = targets[i]['boxes'][0]
                pred_bbox = torch.tensor(boxes[max_score_index]).reshape(4)
                plot_keypoints(image, actual_keypoints, pred_keypoints, actual_bbox, pred_bbox, args.dest_dir, targets[i]['img_name'])
                print(f"model {targets[i]['img_name']} written successfully")
                return

if __name__ == '__main__':
    main()
