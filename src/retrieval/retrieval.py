##Demo: python .\src\retrieval\retrieval.py --model .\src\retrieval\akd_15_08-07-2024_00.pth -i .\src\retrieval\images\pexels-mikebirdy-4639907.jpg --images .\src\dataset\output\images\clean\ --json .\src\retrieval\dataset.json



import argparse
import json
from torchvision.datasets import CocoDetection
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from utils import keypoint_similarity, Inference

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
    
    print(oks)
    higher_score = max(oks.items(),key=lambda x: torch.max(x[1]).item())
    print(higher_score)
    higher_score_path = image_folder+higher_score[0]

    plot_images(image_path,higher_score_path,higher_score)
    
if __name__ == '__main__':
    main()