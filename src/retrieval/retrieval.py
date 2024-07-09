##Demo: python .\src\retrieval\retrieval.py --model .\src\retrieval\akd_15_08-07-2024_00.pth -i .\src\retrieval\images\pexels-mikebirdy-4639907.jpg --images .\src\dataset\output\images\clean\ --json .\src\retrieval\dataset.json



import argparse
import os
import json
from torchvision.datasets import CocoDetection
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from utils import keypoint_similarity
#jeep_192.jpg pexels-mikebirdy-4639907.jpg
# pexels-photo-7808349.jpeg 
NUM_KPT = 20

class COCODataset(CocoDetection):
    def __getitem__(self, idx):
        img, target = super(COCODataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        bbox_yxhw = torch.tensor(anno[0]['boxes']).reshape(1, 4)
        imgs = self.coco.loadImgs(ids=[img_id])

        target = {
            'img_name': imgs[0]['file_name'],
            'boxes': torchvision.ops.box_convert(bbox_yxhw, 'xywh', 'xyxy'),
            'labels': torch.tensor([1 for _ in range(1)], dtype=torch.int64),
            'keypoints': torch.tensor(anno[0]['keypoints']).reshape(1, -1, 3),
        }
        return img, target


def get_transform():
    transforms = [T.ToTensor()]
    return torchvision.transforms.Compose(transforms)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),  # Converti l'immagine in tensore
    ])
    image = transform(image)
    return image


class Inference:
    """
    Inference class to make prediction with a given pre-trained model over a given image.
    Params:
        image: Preprocessed image
        model: Pre-trained model
    """
    def __init__(self,image,model):
        self.image = image
        self.model = model

    def get_kpts(self, is_th = False, th = 50):
        """
        Make predictions with given image and return a keypoint tensor with higher scores.
        Params: 
            is_th: Boolean variable, if True it makes thresholding otherwise not.
            th: Threshold value.

        Return:
            Keypoint tensor with shape [20,3].
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model([self.image])
        self.boxes = predictions[0]['boxes'].cpu().detach().numpy()
        self.keypoints = predictions[0]['keypoints'].cpu().detach().numpy()
        self.scores = predictions[0]['scores'].cpu().detach().numpy()
        self.keypoint_scores = predictions[0]['keypoints_scores'].cpu().detach().numpy()
        # print(predictions[0]['keypoints_scores'])
        kpts = self.keypoints[self.scores.argmax()]
        if is_th:
            kpts_th = self.thresholding(th)
            kpts = kpts_th
        return kpts

    
    def thresholding(self,th):
        """
        Threshold function. 
        If scores is below a given thershold value, set the corresponding element in the keypoints tensor to zero.
        Params:
            th: Threshold value.
        Return:
            Thresholded keypoint tensor. 
        """
        scores = self.keypoint_scores[self.scores.argmax()]
        kpts = self.keypoints[self.scores.argmax()]
        for i in range(0,len(scores)):
            # print(scores[i])
            if scores[i] < th:
                kpts[i] = torch.tensor([0.0, 0.0, 0.0])
        return kpts
    
    def get_area(self):
        """
        Get H,W of higher score bbox for OKS evaluation.
        Return:
            A tensor with [2] shape.
        """
        bbox = torch.tensor(self.boxes[self.scores.argmax()]).reshape(4)
        return bbox[-2:]
    
    def plot_keypoint(self):
        plt.imshow(self.image.permute(1,2,0).cpu().numpy())
        ax = plt.gca()
        max_score_index = self.scores.argmax()
        keypoints = self.keypoints[max_score_index]
        bbox = torch.tensor(self.boxes[max_score_index]).reshape(4)

        ## Draws predicted bbox (red)
        x1, y1, w, h = torchvision.ops.box_convert(bbox, 'xyxy', 'xywh')
        rect = plt.Rectangle((x1, y1), w, h, fill=False, color='red')
        ax.add_patch(rect)

        ## Draws predicted keypoints (green)
        for j, keypoint in enumerate(keypoints):
            x, y, v = keypoint
            circle = plt.Circle((x, y), 2, color='red', fill=True)
            ax.add_patch(circle)

        plt.axis('off')
        plt.show()

def oks_reshape(kpts):
    """
    Reshape the tensor to [M, #kpts, 3] shape.

    Params:
        kpts: Either ground_truth keypoints or retrieval keypoint to reshape.
    Return:
        Reshaped tensor
    """
    
    kpts = kpts.reshape(1,NUM_KPT,3)
    return torch.tensor(kpts, dtype=torch.float32)

def plot_images(image_path,higher_score_path, score):
    retrival_image = plt.imread(higher_score_path)
    original_image = plt.imread(image_path)
    plt.figure(figsize=(15, 8)) 
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.title(f'Image:{score[0]} score: {score[1]}',fontsize=16)
    plt.imshow(retrival_image)
    plt.show()

def get_tensors(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    
    data = {key: [torch.tensor(item) for item in value] for key, value in data.items()}
    
    return data
def main():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--model', type=str, help='Path to .pth model file.')
    parser.add_argument('-i', dest='retrieval_image', type=str, help='Path to retrival image')
    parser.add_argument('--images', dest='folder_images', type=str, help='Path to retrival image')
    parser.add_argument('--json', dest='json_path', type=str, help='Path to retrival image')


    args = parser.parse_args()
    device = torch.device('cpu')
    image_path = args.retrieval_image
    image = preprocess_image(image_path)
    model = keypointrcnn_resnet50_fpn(num_keypoints=NUM_KPT)
    model.load_state_dict(torch.load(args.model, map_location=device))

    ## Do inference over the retrieval image
    retrieval_image = Inference(image,model)
    retrieval_kpts = retrieval_image.get_kpts()
    pd_kpts = oks_reshape(retrieval_kpts)

    batch_image = 0
    image_folder = args.folder_images
    
    KPTS_OKS_SIGMAS_UNIF = torch.ones(NUM_KPT)/NUM_KPT ## Guardare sta roba
    # oks_tensor = {}
    oks = {}
    oks_tensor = get_tensors(args.json_path)
    # print(len(oks_tensor))
    for filename,[gt_kpts,area] in oks_tensor.items():
        similarity = keypoint_similarity(gt_kpts,pd_kpts,KPTS_OKS_SIGMAS_UNIF, area)
        oks[filename] = similarity
        batch_image+=1
        print(f"Images left: {batch_image}")
    


    ## Keypoints eval for the first 'batch_image' images
    # for filename in os.listdir(image_folder):
    #     # if batch_image == 0:
    #     #     break
    #     file_path = os.path.join(image_folder, filename)
    #     ## Do inference over gt_image 
    #     image = preprocess_image(file_path)
    #     gt_image = Inference(image,model)
    #     gt_kpts = gt_image.get_kpts()
    #     # print(gt_kpts)
    #     gt_kpts = oks_reshape(gt_kpts)
    #     area = gt_image.get_area()
    #     similarity = keypoint_similarity(gt_kpts,pd_kpts,KPTS_OKS_SIGMAS_UNIF, area)
    #     oks[filename] = similarity
    #     batch_image+=1
    #     print(f"Images left: {batch_image}")
    
    print(oks)
    higher_score = max(oks.items(),key=lambda x: torch.max(x[1]).item())
    print(higher_score)
    higher_score_path = image_folder+higher_score[0]

    plot_images(image_path,higher_score_path,higher_score)
    
if __name__ == '__main__':
    main()