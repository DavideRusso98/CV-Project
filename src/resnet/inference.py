import argparse
import os

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import keypointrcnn_resnet50_fpn

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

# Preprocessing image function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),  # Converti l'immagine in tensore
    ])
    image = transform(image)
    return image


def plot_keypoints(image, keypoints, scores, bbox, args, threshold=13):  # Soglia impostata a 0.5
    ax = plt.gca()

    x1, y1, w, h = torchvision.ops.box_convert(bbox[0], 'xyxy', 'cxcywh')
    rect = plt.Rectangle((x1, y1), w, h, fill=False, color='red')
    ax.add_patch(rect)

    for j, keypoint in enumerate(keypoints[0]):
        x, y, v = keypoint
        if v > 0 and scores[0][j] > threshold:  # 0 perchè guardiamo solo alla bbox della immagine
            circle = plt.Circle((x, y), 2, color='red', fill=True)
            ax.add_patch(circle)

    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    image_name = 'pred_' + os.path.basename(args.image)
    plt.savefig(os.path.join(args.dest_dir, image_name))


def plot_predictions(image, predictions):
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax = plt.gca()
    boxes = predictions[0]['boxes'].cpu()
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='blue')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('model', type=str, help='Path to .pth model file.')
    parser.add_argument('image', type=str, help='Path to test image')
    parser.add_argument('--out', '-o', dest='dest_dir', type=str, help='Path to output_image')
    parser.add_argument('--images', dest='image_folder', type=str, help='Path to image folder')
    parser.add_argument('--coco', '-c', dest='coco', type=str, help='Path to .json coco annotations file')

    args = parser.parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)

    model = keypointrcnn_resnet50_fpn(num_keypoints=20)
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    model.eval()
    image = preprocess_image(args.image)

    with torch.no_grad():
        prediction = model([image])

    image = image.cpu()
    keypoints = prediction[0]['keypoints'].cpu()
    scores = prediction[0]['keypoints_scores'].cpu()
    bbox = prediction[0]['boxes'].cpu()

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

    plot_keypoints(image, keypoints, scores, bbox, args, threshold=0)


if __name__ == '__main__':
    main()
