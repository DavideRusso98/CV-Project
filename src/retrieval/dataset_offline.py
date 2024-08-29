
##  python .\src\retrieval\dataset_offline.py --model .\src\retrieval\akd_15_08-07-2024_00.pth --images .\src\dataset\output\images\clean\ --json .\src\retrieval\dataset.json
import os,argparse,torch,json
from retrieval import Inference, get_transform, preprocess_image,oks_reshape
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor, KeypointRCNN
from components import AutomotiveKeypointDetector
import torchvision


NUM_KPT = 20
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
def write_json(oks,path):
    oks =  {key: [tensor.tolist() for tensor in value] for key, value in oks.items()}
    json_data = json.dumps(oks, indent=4)
    with open(path, 'w') as json_file:
        json_file.write(json_data)
def main():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--model', type=str, help='Path to .pth model file.')
    parser.add_argument('--images', dest='folder_images', type=str, help='Path to retrival image')
    parser.add_argument('--json', dest='json_path', type=str, help='Path to retrival image')
    
    args = parser.parse_args()
    image_folder = args.folder_images
    device = torch.device('cpu')
    dilation = 2
    kh_depth = 6
    model = AutomotiveKeypointDetector(kh_depth=kh_depth, dilation=dilation)
    model.load_state_dict(torch.load(args.model, map_location=device))

    batch_image = 0
    oks = {}

    for filename in os.listdir(image_folder):
            # if batch_image == 0:
            #     break
            kpts = []
            file_path = os.path.join(image_folder, filename)
            ## Do inference over gt_image 
            image = preprocess_image(file_path)
            gt_image = Inference(image,model)
            ## Keypoint eval, True for thresholding
            gt_kpts = gt_image.get_kpts(True)
            gt_kpts = oks_reshape(gt_kpts)
            kpts.append(gt_kpts)
            ## Area eval
            area = gt_image.get_area()
            kpts.append(area)
            oks[filename] = kpts

            batch_image+=1
            print(f"Images left: {batch_image}")
    # print(oks)
    write_json(oks,args.json_path)
if __name__ == '__main__':
    main()