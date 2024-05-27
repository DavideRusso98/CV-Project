import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os
import cv2
import argparse

def show_keypoints(im, im_name, keypoints):
    """
    Show keypoints on an image.

    :param im: The image to display.
    :param keypoints: The keypoint coordinates in coco format [(x, y, visibility)]
    :return: None

    """
    plt.ion()  # interactive mode
    x_vector = keypoints[0::3]
    y_vector = keypoints[1::3]
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title(im_name)
    plt.scatter(x_vector, y_vector, 30, c="r", marker="+")
    plt.imshow(im, cmap="gray")  # plot image
    plt.show(block=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', nargs='?', help="car model name")
    parser.add_argument('output_dir', nargs='?', help="Path to output/ dir")
    args = parser.parse_args()

    ### load coco .json
    coco_annotations = os.path.join(args.output_dir, 'coco_annotations.json')
    coco = COCO(coco_annotations)

    ### load image
    im = cv2.imread(os.path.join(args.output_dir, "images", f"{args.model_name}.jpg"))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ### load annotations
    annIds = coco.getAnnIds(imgIds=[args.model_name])
    anns = coco.loadAnns(annIds)
    keypoints = anns[0]['keypoints']


    show_keypoints(im, args.model_name, keypoints)



if __name__ == '__main__':
    main()