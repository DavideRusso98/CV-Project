import torch
from pycocotools.coco import COCO

def keypointv_to_featurev(keypoint_vector):
    feature_v = []
    semantic = 0
    i = 0
    while i < len(keypoint_vector):
        x, y, vis = keypoint_vector[i], keypoint_vector[i+1], keypoint_vector[i+2]
        feature_v.extend([x, y, semantic])
        semantic += 1
        i += 3
    return torch.tensor(feature_v)


IMAGES_PATH = "/mnt/c/Users/alesv/PycharmProjects/CV-Project/src/dataset/output/images/clean/tesla/"
ANNOTATION_PATH = "/mnt/c/Users/alesv/PycharmProjects/CV-Project/src/dataset/output/coco_annotations.json"

coco = COCO(ANNOTATION_PATH)
annIds = coco.getAnnIds(imgIds=[2, 5, 9])
anns = coco.loadAnns(annIds)

tens1 = keypointv_to_featurev(anns[0]['keypoints'])
tens2 = keypointv_to_featurev(anns[1]['keypoints'])
tens3 = keypointv_to_featurev(anns[2]['keypoints'])

cos_sim = torch.nn.CosineSimilarity(dim=0)

tens1_tens2_diff = cos_sim(tens1, tens2)
print(f"tens1_tens2_diff: {tens1_tens2_diff}")

tens1_tens3_diff = cos_sim(tens1, tens3)
print(f"tens1_tens3_diff: {tens1_tens3_diff}")

tens2_tens3_diff = cos_sim(tens2, tens3)
print(f"tens2_tens3_diff: {tens2_tens3_diff}")