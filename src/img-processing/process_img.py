import cv2
import numpy as np
import os

input_folder = './src/img-processing/car/output_images/no_bg'
output_folder = './src/img-processing/car/output_images/denoised'

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    image = denoise_image(image)
    image = enhance_contrast(image)
    image = sharpen_image(image)
    image = normalize_image(image)
    
    cv2.imwrite(output_path, image)

def denoise_image(image):
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    #denoised = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised

def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = enhanced_gray
    enhanced_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return enhanced_image

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def normalize_image(image):
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            preprocess_image(input_path, output_path)
            print(f'Processed {input_path} and saved to {output_path}')

process_images(input_folder, output_folder)
