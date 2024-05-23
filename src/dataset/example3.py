import bpy
import json

# Funzione per raccogliere keypoint
def get_keypoints():
    keypoints = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'EMPTY':  # Consideriamo solo gli oggetti di tipo 'Empty'
            keypoints.append({
                'name': obj.name,
                'location': [obj.location.x, obj.location.y, obj.location.z]
            })
    return keypoints

# Raccogli i keypoint
keypoints = get_keypoints()

# Percorso del file di output
output_file = "C://Users/Alesv/output.json"

# Scrivi i keypoint in un file JSON
with open(output_file, 'w') as f:
    json.dump(keypoints, f, indent=4)