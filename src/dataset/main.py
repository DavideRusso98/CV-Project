import blenderproc as bproc
import uuid
import argparse
from enum import Enum
from dataclasses import dataclass
import json
import os
import numpy as np
import bpy
import cv2
import datetime

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

class KEYPOINT_TYPE(Enum):
    TYPE_2D = 'TYPE_2D'
    TYPE_3D = 'TYPE_3D'

@dataclass
class Keypoint:
    type: KEYPOINT_TYPE
    semantic: str
    location: np.array
    def __init__(self, type, semantic, location):
        self.type = type
        self.semantic = semantic
        self.location = location
    def to_serializable_dict(self):
        return {'type': str(self.type.value), 'semantic': self.semantic, 'location': self.location.tolist()}

def keypoint_3d_to_2d(keypoint3d: Keypoint, frame_id) -> Keypoint:
    """
    Convert a 3D keypoint to a 2D keypoint representation.

    :param keypoint3d: The 3D keypoint to be converted.
    :type keypoint3d: Keypoint

    :return: The 2D keypoint representation of the input 3D keypoint.
    :rtype: Keypoint
    """
    location2d = bproc.camera.project_points(np.array([keypoint3d.location]), frame_id)[0]
    return Keypoint(KEYPOINT_TYPE.TYPE_2D, keypoint3d.semantic, location2d)

def extract_3d_keypoints(scene) -> [Keypoint]:
    """

    This method `extract_3d_keypoints` extracts 3D keypoints from the given scene.

    :param scene: A list of objects in the scene.
    :return: A list of 3D keypoints.

    """
    keypoints3d = []
    for obj in scene:
        o = obj.blender_obj
        if o.type == 'EMPTY':  ## filters out 'MESH' type
            keypoints3d.append(Keypoint(KEYPOINT_TYPE.TYPE_3D, o.name, np.array([o.location.x, o.location.y, o.location.z])))
    return keypoints3d

def init_coco() -> dict:
    """
    Initializes a Coco annotations dictionary for an Automotive KeyPoints Dataset.

    :return: Coco annotations dictionary
    :rtype: dict
    """
    coco_annotations = {
        "info": {
            "description": "Automotive KeyPoints Dataset",
            "version": "0.1",
            "year": 2024,
            "contributor": "GRUPPO14_Prini_Russo_Valenza",
            "date_created": datetime.datetime.now().isoformat()
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }
    return coco_annotations

def coco_write_categories(keypoints2d_list, coco):
    """
    This method is deprecated. Now we are using a single category "car" ("coco_default_category" method)

    :param keypoints2d_list A list of keypoints
    :param coco A dictionary representing a COCO dataset

    """
    semantic_to_category_id = {}
    category_id_counter = 1
    for item in keypoints2d_list:
        semantic = item["semantic"]
        if semantic not in semantic_to_category_id:
            category = {
                "id": category_id_counter,
                "name": semantic,
                "supercategory": "car",
                "keypoints": [semantic], ##TODO: capire bene cosa inserire qui e se semantic è ok
                "skeleton": []
            }
            semantic_to_category_id[semantic] = category_id_counter
            coco["categories"].append(category)
            category_id_counter += 1

def coco_append_image(coco, image_id):
    image_info = {
        "id": image_id,
        "file_name": f"{image_id}.jpg",
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH
    }
    coco["images"].append(image_info)

def coco_append_keypoints(coco, keypoints, image_id, category_id):
    annotation = {
        "id": uuid.uuid4(),
        "num_keypoints": len(keypoints) / 3,
        "keypoints": keypoints,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
    }
    coco["annotations"].append(annotation)

def coco_default_category(coco):
    """
    Add a default category with id 1 to the given COCO object.

    :param coco: The COCO object to add the category to.
    """
    category = {
            "supercategory": "vehicle",
            "id": 1,
            "name": "car",
            "keypoints": [],
            "skeleton": []
        }
    coco["categories"].append(category)

def write_to_json(coco, output_dir):
    """
    Write the given Coco dataset to a JSON file.

    :param coco: A dictionary representing the Coco dataset.
    :param output_dir: The directory path where the JSON file will be saved.
    :return: None

    This method takes a Coco dataset dictionary and a directory path as input parameters.
    It writes the Coco dataset to a JSON file with the name "coco_annotations.json" in the specified directory.
    The Coco dataset dictionary is first converted to a pretty-printed JSON string using the `json.dumps` method.
    Then, the JSON string is written to the file using the `write` method of a file object.

    Example usage:
    ```python
    coco = {...}  # Coco dataset
    output_dir = "/path/to/directory/"
    write_to_json(coco, output_dir)
    ```
    """
    filename = output_dir + "coco_annotations.json"
    with open(filename, "w") as f:
        pretty = json.dumps(coco, indent=4)
        f.write(pretty)

def load_transf_matrix_list() -> list:
    """
    Load transformation matrix list from a JSON file.

    :return: The first two transformation matrices in the list.
    """
    with open('./src/dataset/transforms_train.json', 'r') as file:
        json_data = file.read()
    data = json.loads(json_data)
    trasformation_matrix_list = [frame['transform_matrix'] for frame in data['frames']]
    return trasformation_matrix_list[:2]

def init_light_and_resolution(image_width, image_height):
    bproc.camera.set_resolution(image_width, image_height)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

def write_image_to_file(output_dir, image_id, image_rgb):
    """
    Write the given RGB image to a file in the specified output directory.

    :param output_dir: The directory where the image file should be saved.
    :param image_id: The unique identifier for the image.
    :param image_rgb: The RGB image to be written.
    :return: None
    """
    color_bgr = image_rgb.copy()
    color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]
    target_path = os.path.join(output_dir, f"images/{image_id}.jpg")
    cv2.imwrite(target_path, color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', nargs='?', help="Path to the 3d model in .ply format")
    parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved")
    args = parser.parse_args()

    bproc.init()
    coco = init_coco()
    coco_default_category(coco)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    for matrix in load_transf_matrix_list():
        bproc.camera.add_camera_pose(matrix)

    scene = bproc.loader.load_blend(args.scene)
    keypoints3d = extract_3d_keypoints(scene)

    init_light_and_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)
    data = bproc.renderer.render()

    colors = data["colors"]
    model_name = "tesla"

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):

        ### Extract image, saves it to file and populate coco_annotation's "images" section
        image_id = f"{model_name}_{frame}" # image names will be like "testa_01","tesla_02",...
        image_rgb = colors[frame - bpy.context.scene.frame_start]
        write_image_to_file(args.output_dir, image_id, image_rgb)
        coco_append_image(coco, image_id)

        ### Extract keypoints and saves them into coco_annotation's "annotation" section
        keypoints2d = [keypoint_3d_to_2d(keypoint, frame) for keypoint in keypoints3d]
        coco_keypoints = (np.array([np.append(key2d.location, 2) for key2d in keypoints2d])## np.append: 2 is for visibility
                          .flatten().tolist()) # flatten because coco keypoints should be a single list of size N*3
        coco_append_keypoints(coco, coco_keypoints, image_id, 1)

    write_to_json(coco, args.output_dir)

if __name__ == "__main__":
    main()