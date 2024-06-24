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
import random

from blenderproc.python.writer.CocoWriterUtility import _CocoWriterUtility

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

"""
README:
While rendering multiple models an ERROR occurs:
ERROR (bke.lib_id_delete): source/blender/blenkernel/intern/lib_id_delete.c:357 id_delete: Deleting IMRender Result which still has 1 users (including 0 'extra' shallow users)

Ignore it, the script is working. 
See issue: https://github.com/DLR-RM/BlenderProc/issues/997
"""


class KEYPOINT_TYPE(Enum):
    TYPE_2D = 'TYPE_2D'
    TYPE_3D = 'TYPE_3D'

semantic_keypoints = [
    "wheel_FL",
    "wheel_FR",
    "wheel_RL",
    "wheel_RR",
    "door_FL",
    "door_FR",
    "door_RL",
    "door_RR",
    "headlight_FL",
    "headlight_FR",
    "headlight_RL",
    "headlight_RR",
    "windshield_FRONT",
    "windshield_REAR",
    "bumper_FRONT",
    "bumper_REAR",
    "license_plate",
    "mirror_LEFT",
    "mirror_RIGHT",
    "roof",
]

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

def extract_3d_keypoints(scene):
    keypoints3d_dict = dict()
    for obj in scene:
        o = obj.blender_obj
        if o.type == 'EMPTY':
            keypoints3d_dict[o.name] = Keypoint(KEYPOINT_TYPE.TYPE_3D, o.name, np.array([o.location.x, o.location.y, o.location.z]))
    return keypoints3d_dict

def sorting_keypoints(keypoints3d) -> [Keypoint]:
    """
    Sorting function using `semantic_keypoints` list.
    
    :param keypoints3d: list of Keypoints
    :return: Sorted list 
    """
    order_dict = {key: index for index, key in enumerate(semantic_keypoints)}
    #print(order_dict)
    sorted_keypoints = sorted(keypoints3d, key=lambda x: order_dict.get(x.semantic, float('inf')))
    return sorted_keypoints

def compute_coco_keypoints(keypoints3d_dict, frame):
    coco_keypoints = []
    num_keypoints = 0
    for semantic in semantic_keypoints:
        if semantic not in keypoints3d_dict:
            coco_keypoints.extend([0, 0, 0])
            continue
        keypoint2d = keypoint_3d_to_2d(keypoints3d_dict[semantic], frame)
        coco_keypoints.extend(keypoint2d.location)
        coco_keypoints.append(2) # visibility
        num_keypoints+=1
    return (coco_keypoints, num_keypoints)

def getChildren(myObject): 
    """
    Return children of a given blender object

    :param myObject: Blender object 
    :return: Object's children 
    """
    children = [] 
    for ob in bpy.data.objects: 
        if ob.parent == myObject: 
            children.append(ob) 
    return children

def extract_bbox(scene):
    """ 
    Function `extract_bbox` extracts bbox coordinates from a give scene

    :param scene: A list of objects in the scene.
    :return: A list bbox_coordinates [xmin,ymin,xmax,ymax].
    """
    bounding_boxes = []
    for obj in scene:
        o = obj.blender_obj
        if o.type == "EMPTY":
            #print(o.name, o.type)
            child = getChildren(o)
            bbox = child[0].bound_box
            xmin = min([v[0] for v in bbox])
            ymin = min([v[1] for v in bbox])
            xmax = max([v[0] for v in bbox])
            ymax = max([v[1] for v in bbox])
        
            if([xmin, ymin, xmax, ymax] != [0.0, 0.0, 0.0, 0.0]):
                bounding_boxes.append([xmin, ymin, xmax, ymax])
        #print(bounding_boxes)
    return bounding_boxes

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
                "keypoints": [semantic_keypoints], ##TODO: capire bene cosa inserire qui e se semantic è ok
                "skeleton": []
            }
            semantic_to_category_id[semantic] = category_id_counter
            coco["categories"].append(category)
            category_id_counter += 1

def coco_append_image(coco, image_filename, image_id) -> int:
    image_info = {
        "id": image_id,
        "file_name": f"{image_filename}.jpg",
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH
    }
    coco["images"].append(image_info)

def coco_append_keypoints(coco, keypoints, num_keypoints, bbox, image_id, category_id):
    annotation = {
        "id": str(uuid.uuid4()),
        "num_keypoints": num_keypoints,
        "keypoints": keypoints,
        "boxes": bbox,
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
            "keypoints": [semantic_keypoints],
            "skeleton": []
        }
    coco["categories"].append(category)

def write_to_json(coco, output_dir):
    """
    Write the given Coco dataset to a JSON file.

    :param coco: A dictionary representing the Coco dataset.
    :param output_dir: The directory path where the JSON file will be saved.
    :return: None

    This methoblenderproc visd takes a Coco dataset dictionary and a directory path as input parameters.
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
        pretty = json.dumps(coco, indent=4,separators=(',', ': '))
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
    #return trasformation_matrix_list[]
    return [trasformation_matrix_list[1], trasformation_matrix_list[2], trasformation_matrix_list[60]] ## todo: uncomment this in prod

def init_light_and_resolution(image_width, image_height):
    bproc.camera.set_resolution(image_width, image_height)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(10000)
    return light

def write_image_to_file(output_dir, image_id, image_bgr, bbox):
    ### Write rendered image without bbox and keypoints
    target_path = os.path.join(output_dir, f"images/clean/{image_id}.jpg")
    cv2.imwrite(target_path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    ###Write same rendered image with bbox and keypoints (keypoints are still in todo)
    x, y, w, h = bbox
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    target_path = os.path.join(output_dir, f"images/annotated/{image_id}.jpg")
    cv2.imwrite(target_path, image_bgr,[int(cv2.IMWRITE_JPEG_QUALITY), 95])

def add_camera_poses(matrix_list):
    for matrix in matrix_list:
        bproc.camera.add_camera_pose(matrix)

def return_boundingbox_from_segmap(segmap):
    binary_mask = np.where(segmap != 0, 1, 0)
    return _CocoWriterUtility.bbox_from_binary_mask(binary_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenes_dir', nargs='?', help="Path to the 3d model directory")
    parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved")
    args = parser.parse_args()

    bproc.init()
    coco = init_coco()
    coco_default_category(coco)
    os.makedirs(os.path.join(args.output_dir, 'images/clean'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images/annotated'), exist_ok=True)

    camera_poses = load_transf_matrix_list()
    scenes_paths = [args.scenes_dir + scene_filename for scene_filename in os.listdir(args.scenes_dir)]
    model_names = [scene_filename.split('_')[0] for scene_filename in os.listdir(args.scenes_dir)]
    model_list = list(zip(scenes_paths, model_names))
    #model_list = model_list[:2] ## TODO: remove

    image_id = 0
    for model_path, model_name in model_list:
        add_camera_poses(camera_poses)
        init_light_and_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)
        bproc.renderer.set_world_background([1,1,1]) # this should set the background to black
        scene = bproc.loader.load_blend(model_path)
        for j, obj in enumerate(scene):
            obj.set_cp("category_id", j + 1)

        keypoints3d_dict = extract_3d_keypoints(scene)
        #keypoints3d = sorting_keypoints(keypoints3d_dict)

        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name", "class"])
        data = bproc.renderer.render()
        colors = data["colors"]

        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            image_filename = f"{model_name}_{frame}"
            image_rgb = colors[frame - bpy.context.scene.frame_start]
            color_bgr = image_rgb.copy()
            color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]
            coco_bbox = return_boundingbox_from_segmap(data["instance_segmaps"][frame])
            write_image_to_file(args.output_dir, image_filename, color_bgr, coco_bbox)
            coco_append_image(coco, image_filename, image_id)
            coco_keypoints, num_keypoints = compute_coco_keypoints(keypoints3d_dict, frame)
            coco_append_keypoints(coco, coco_keypoints, num_keypoints, coco_bbox, image_id, 1)
            image_id+=1

        ## cleans the scene before loading the next model
        bproc.clean_up()

    write_to_json(coco, args.output_dir)

if __name__ == "__main__":
    main()