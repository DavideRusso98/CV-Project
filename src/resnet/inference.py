import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# Loading pre-trained model
model = keypointrcnn_resnet50_fpn(pretrained=False, num_classes=2, num_keypoints=20)
model.load_state_dict(torch.load('/mnt/c/Users/alesv/PycharmProjects/CV-Project/our_super_model_trained.pth'))
model.eval()

# Preprocessing image function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),  # Converti l'immagine in tensore
    ])
    image = transform(image)
    return image

# Percorso dell'immagine
image_path = "/mnt/c/Users/alesv/PycharmProjects/CV-Project/src/dataset/output/images/clean/alfa_0.jpg"
image = preprocess_image(image_path)

# Effettua l'inferenza
with torch.no_grad():
    prediction = model([image])
def plot_keypoints(image, keypoints, scores, threshold=0.5):
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax = plt.gca()
    for i, keypoint_set in enumerate(keypoints):
        if scores[i] > threshold:
            for keypoint in keypoint_set:
                x, y, v = keypoint
                print(keypoint)
                if v > 0:  # keypoint visibile
                    circle = plt.Circle((x, y), 2, color='red', fill=True)
                    ax.add_patch(circle)
    plt.axis('off')
    plt.show()

# Visualizza i risultati
image = image.cpu()
keypoints = prediction[0]['keypoints'].cpu()
scores = prediction[0]['scores'].cpu()
print(keypoints)
print(scores)
plot_keypoints(image, keypoints, scores)
