import datetime

# Crea un dataset vuoto
data = {
    "info": {
        "description": "KeyPoints Dataset",
        "version": "1.0",
        "year": 2024,
        "contributor": "AI Assistant",
        "date_created": datetime.datetime.now().isoformat()
    },
    "images": [],
    "annotations": [],
    "categories": [],
}

# Definisci la tua categoria (può essere un'azione, un oggetto, ecc...)
category = {
    "id": 1,
    "name": "person",
    "supercategory": "person",
    "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                  "left_wrist", "right_wrist", "left_hip", "right_hip",
                  "left_knee", "right_knee", "left_ankle", "right_ankle"],  # Gli keypoints che vuoi annotare
    "skeleton": [[0, 1], [0, 2], ...]  # Le connessioni tra i keypoints come coppie di indici
    # (ad esempio, un osso può andare dal "nose" - 0, a "left_eye" - 1)
}

# Aggiungi la categoria al tuo dataset
data['categories'].append(category)

# Hai bisogno di specificare la tua immagine
image = {
    "id": 1,
    "width": 640,
    "height": 480,
    "file_name": "mia_immagine.jpg",
    "license": "",
    "date_captured": datetime.datetime.now().isoformat()
}

# Aggiungi la tua immagine al tuo dataset
data['images'].append(image)

# Definisci la tua annotazione con i keypoints che hai rilevato.
# I keypoints devono essere rappresentati come una lista unidimensionale con una lunghezza di
# num_keypoints * 3. Ogni gruppo di tre valori rappresenta [x, y, visibilità].
annotation = {
    "id": 1,
    "image_id": 1,  # Questo deve corrispondere all'ID dell'immagine corrispondente
    "category_id": 1,  # Questo deve corrispondere all'ID della categoria

    "keypoints": [350, 160, 2, 360, 150, 2, ..., 100, 200, 2],  # Inserisci i tuoi keypoints qui.
    # Nota che la visibilità è spesso: 0=invisibile,
    # 1=visibile ma occluso, 2=visibile

    "num_keypoints": 5,  # Il numero totale di keypoints
    "bbox": [320, 120, 640, 480],  # La bounding box del tuo oggetto come [top left x, top left y, width, height]
    "iscrowd": 0,  # Questo dovrebbe essere 0 se la tua bounding box racchiude solo una singola istanza
}

# Aggiungi l'annotazione al tuo dataset
data['annotations'].append(annotation)

# Alla fine, dovresti salvare il tuo file di annotazioni in formato JSON
import json

with open('mia_annotazione.json', 'w') as output_json_file:
    json.dump(data, output_json_file)
