import blenderproc as bproc
import argparse
from enum import Enum
from dataclasses import dataclass
import json
import os

import numpy as np
import bpy
import cv2
import datetime

from blenderproc.python.writer.CocoWriterUtility import _CocoWriterUtility
from mathutils import Vector
from matplotlib import pyplot as plt

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


keypoint_semantic_list = [
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


def is_visible(keypoint3d, frame_id):

    current_camera_pose = bproc.camera.get_camera_pose(frame_id)
    camera_location = Vector(get_camera_location(current_camera_pose))

    keypoint_location = Vector(keypoint3d.location)
    direction = keypoint_location - camera_location
    threshold = 0.2

    ## cast a ray from the camera in the given direction, returns hit_location if hits something
    hit, hit_location, _, _, _, _ = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph,
        origin=camera_location + direction * 0.0001, direction=direction)

    ## computes the L2 distance between the point hit by the ray and keypoint actual location
    distance = np.linalg.norm(hit_location - keypoint_location)
    print(f"frame: {frame_id} distance: {distance}")
    #print(f"{keypoint3d.semantic} is visible: {distance < threshold}, distance: {distance}")
    if not hit:
        return False
    return distance < threshold


def keypoint_3d_to_2d(keypoint3d: Keypoint, frame_id):
    if not is_visible(keypoint3d, frame_id):
        return False, [0, 0, 0]
    projected_points = bproc.camera.project_points(np.array([keypoint3d.location]), frame_id)
    x, y = projected_points[0]
    return True, [x, y, 2]


def extract_3d_keypoints(scene):
    keypoints3d_dict = dict()
    for obj in scene:
        o = obj.blender_obj
        if o.type == 'EMPTY':
            keypoints3d_dict[o.name] = Keypoint(KEYPOINT_TYPE.TYPE_3D, o.name,
                                                np.array([o.location.x, o.location.y, o.location.z]))
    return keypoints3d_dict


def compute_coco_keypoints(keypoints3d_dict, frame):
    coco_keypoints = []
    num_keypoints = 0
    for keypoint_name in keypoint_semantic_list:
        if keypoint_name not in keypoints3d_dict:
            coco_keypoints.extend([0, 0, 0])
            continue
        visibility, coco_keypoint = keypoint_3d_to_2d(keypoints3d_dict[keypoint_name], frame)
        coco_keypoints.extend(coco_keypoint)
        if visibility:
            num_keypoints += 1
    return coco_keypoints, num_keypoints


def load_coco_annotations():
    with open('./src/dataset/output/coco_annotations.json', 'r') as file:
        json_data = file.read()
    return json.loads(json_data)


def init_coco() -> dict:
    coco_annotations = {
        "info": {
            "description": "Automotive KeyPoints Dataset",
            "version": "0.1",
            "year": 2024,
            "contributor": "GRUPPO14_Prini_Russo_Valenza",
            "date_created": datetime.datetime.now().isoformat()
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    coco_default_category(coco_annotations)
    return coco_annotations


def coco_append_image(coco, image_filename, image_id) -> int:
    image_info = {
        "id": image_id,
        "file_name": f"{image_filename}.jpg",
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH
    }
    coco["images"].append(image_info)


def coco_append_keypoints(coco, keypoints, num_keypoints, bbox, segmentation, image_id, category_id):
    annotation = {
        "id": image_id + 10000,
        "num_keypoints": num_keypoints,
        "keypoints": keypoints,
        "boxes": bbox,
        "segmentation": segmentation,
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
        "keypoints": [keypoint_semantic_list],
        "skeleton": []
    }
    coco["categories"].append(category)


def write_to_json(coco, output_dir):
    filename = output_dir + "coco_annotations.json"
    with open(filename, "w") as f:
        pretty = json.dumps(coco, indent=4, separators=(',', ': '))
        f.write(pretty)


def load_transf_matrix_list() -> list:
    """
    Load transformation matrix list from a JSON file.

    :return: The first two transformation matrices in the list.
    """
    with open('./src/dataset/transforms_train.json', 'r') as file:
        json_data = file.read()
    data = json.loads(json_data)
    transformation_matrix_list = [frame['transform_matrix'] for frame in data['frames']]
    return transformation_matrix_list


def init_light_and_resolution(image_width, image_height):
    bproc.camera.set_resolution(image_width, image_height)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(2000)
    return light


def write_and_annotate_image_to_file(output_dir, image_id, image_bgr, bbox, coco_keypoints):
    x, y, w, h = bbox
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    i = 0
    while i < len(coco_keypoints):
        x, y, v = coco_keypoints[i], coco_keypoints[i + 1], coco_keypoints[i + 2]
        x, y = int(round(x)), int(round(y))
        if v != 0:
            cv2.circle(image_bgr, (x, y), 3, (0, 0, 255), 2, cv2.LINE_4)
        i += 3
    target_path = os.path.join(output_dir, f"images/annotated/{image_id}.jpg")
    cv2.imwrite(target_path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def write_image_to_file(output_dir, image_id, image_bgr):
    target_path = os.path.join(output_dir, f"images/clean/{image_id}.jpg")
    cv2.imwrite(target_path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def get_camera_location(transformation_matrix):
    return transformation_matrix[:3, 3]


def add_camera_poses(matrix_list):
    for matrix in matrix_list:
        bproc.camera.add_camera_pose(matrix)


def bbox_and_segmentation(segmentation):
    binary_mask = np.where(segmentation != 0, 1, 0)
    segmentation = _CocoWriterUtility.binary_mask_to_polygon(binary_mask, tolerance=2)
    bbox = _CocoWriterUtility.bbox_from_binary_mask(binary_mask)
    return bbox, segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenes_dir', nargs='?', help="Path to the 3d model directory")
    parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved")
    parser.add_argument('model_number', type=int, help="0 to 9, model number to render")
    args = parser.parse_args()

    bproc.init()
    image_id = 0
    if args.model_number == 0:
        coco = init_coco()
    else:
        coco = load_coco_annotations()
        image_id = coco["images"][-1]["id"] + 1
    print(f"last saved image_id +1 : {image_id}")
    os.makedirs(os.path.join(args.output_dir, 'images/clean'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images/annotated'), exist_ok=True)

    camera_poses = load_transf_matrix_list()
    scenes_paths = [args.scenes_dir + scene_filename for scene_filename in os.listdir(args.scenes_dir)]
    model_names = [scene_filename.split('_')[0] for scene_filename in os.listdir(args.scenes_dir)]
    model_list = list(zip(scenes_paths, model_names))
    model_list = [model_list[args.model_number]]

    for model_path, model_name in model_list:
        add_camera_poses(camera_poses)
        init_light_and_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)
        bproc.renderer.set_world_background([1, 1, 1])  # this should set the background to black
        scene = bproc.loader.load_blend(model_path)
        for j, obj in enumerate(scene):
            obj.set_cp("category_id", j + 1)
        keypoints3d_dict = extract_3d_keypoints(scene)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name", "class"])
        data = bproc.renderer.render()
        colors = data["colors"]
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            image_filename = f"{model_name}_{frame}"
            image_rgb = colors[frame - bpy.context.scene.frame_start]
            color_bgr = image_rgb.copy()
            color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]
            coco_bbox, coco_segmentation = bbox_and_segmentation(data["instance_segmaps"][frame])
            coco_keypoints, num_keypoints = compute_coco_keypoints(keypoints3d_dict, frame)
            write_image_to_file(args.output_dir, image_filename, color_bgr)
            write_and_annotate_image_to_file(args.output_dir, image_filename, color_bgr, coco_bbox, coco_keypoints)
            coco_append_image(coco, image_filename, image_id)
            coco_append_keypoints(coco, coco_keypoints, num_keypoints, coco_bbox, coco_segmentation, image_id, 1)
            image_id += 1

        ## cleans the scene before loading the next model
        bproc.clean_up()

    write_to_json(coco, args.output_dir)


if __name__ == "__main__":
    main()
