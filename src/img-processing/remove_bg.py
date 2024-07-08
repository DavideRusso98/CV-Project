import cv2
import numpy as np
import os

input_folder = './src/img-processing/car/input_images'
output_folder = './src/img-processing/car/output_images/no_bg'

def resize_image(image_path):
    
    image = cv2.imread(image_path)
    
    h, w = image.shape[:2] # Ottieni le dimensioni dell'immagine 
    ratio = 800 / max(h, w) # Calcola il rapporto tra altezza e larghezza dell'immagine
    
    # Calcola le nuove dimensioni mantenendo le proporzioni
    new_h = int(h * ratio)
    new_w = int(w * ratio)
   
    resized_image = cv2.resize(image, (new_w, new_h)) # Ridimensiona l'immagine con le nuove dimensioni
    final_image = np.zeros((800, 800, 3), dtype=np.uint8) # Crea un'immagine di sfondo 800x800 con colore di sfondo nero
    
    # Centra l'immagine ridimensionata sullo sfondo
    start_y = (800 - new_h) // 2
    start_x = (800 - new_w) // 2

    final_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized_image

    return final_image

def remove_background(image, output_path):
    
    mask = np.zeros(image.shape[:2], np.uint8) # Crea una maschera iniziale per grabCut

    # Crea modelli temporanei che grabCut utilizza internamente
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100) # Definisci un rettangolo che contiene l'oggetto (in pixel)
   
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT) # Applica grabCut

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') # Crea la maschera finale dove i valori sono 0 o 1, per indicare il background o il foreground

    foreground = image * mask2[:, :, np.newaxis] # Applica la maschera per ottenere il foreground

    background = np.full_like(image, (200, 200, 200), dtype=np.uint8) # Crea un'immagine di sfondo bianca

    result = background * (1 - mask2[:, :, np.newaxis]) + foreground # Combina il foreground con il background

    cv2.imwrite(output_path, result)

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            remove_background(resize_image(input_path), output_path)
            print(f'Processed {input_path} and saved to {output_path}')

process_images(input_folder, output_folder)
