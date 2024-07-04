import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import keypointrcnn_resnet50_fpn

root = "/home/riccardo/Scrivania/UNIVERSITA/Computer Vision and Cognitive Systems/CV-Project/src/resnet"
# Loading pre-trained model
model = keypointrcnn_resnet50_fpn(pretrained=False, num_classes=2, num_keypoints=20)
model.load_state_dict(torch.load(f'{root}/automotive_keypoint_detector.pth', map_location=torch.device('cpu')))
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
image_path = "./src/dataset/output/images/clean/alfa_56.jpg"
image = preprocess_image(image_path)

# Effettua l'inferenza
with torch.no_grad():
    prediction = model([image])

def plot_keypoints(image, keypoints, scores, threshold=13):  # Soglia impostata a 0.5
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax = plt.gca()
    for i, keypoint_set in enumerate(keypoints):
        for j, keypoint in enumerate(keypoint_set):
            x, y, v = keypoint
            if v > 0 and scores[0][j] > threshold:  # 0 perchè guardiamo solo alla bbox della immagine
                circle = plt.Circle((x, y), 2, color='red', fill=True)
                ax.add_patch(circle)
    plt.axis('off')
    plt.show()

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

# Visualizza i risultati
image = image.cpu()
# plot_predictions(image, prediction)
keypoints = prediction[0]['keypoints'].cpu()
scores = prediction[0]['keypoints_scores'].cpu()
# print(prediction)

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

plot_keypoints(image, keypoints, scores, mean_score)
