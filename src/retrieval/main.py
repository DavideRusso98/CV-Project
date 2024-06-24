import torch
from pycocotools.coco import COCO


IMAGES_PATH = "/mnt/c/Users/alesv/PycharmProjects/CV-Project/src/dataset/output/images/clean/tesla/"
ANNOTATION_PATH = "/mnt/c/Users/alesv/PycharmProjects/CV-Project/src/dataset/output/coco_annotations.json"

coco = COCO(ANNOTATION_PATH)
annIds = coco.getAnnIds(imgIds=[0, 3, 6])
anns = coco.loadAnns(annIds)

tens1 = torch.tensor(anns[0]['keypoints'])
tens2 = torch.tensor(anns[1]['keypoints'])
tens3 = torch.tensor(anns[2]['keypoints'])

cos_sim = torch.nn.CosineSimilarity(dim=0)

tens1_tens2_diff = cos_sim(tens1, tens2)
print(f"tens1_tens2_diff: {tens1_tens2_diff}")

tens1_tens3_diff = cos_sim(tens1, tens3)
print(f"tens1_tens3_diff: {tens1_tens3_diff}")

tens2_tens3_diff = cos_sim(tens2, tens3)
print(f"tens2_tens3_diff: {tens2_tens3_diff}")