import blenderproc as bproc
import argparse
from enum import Enum
from dataclasses import dataclass
import json
import os
import numpy as np
import bpy
import cv2


transform_matrix = [
                [
                    0.6236465573310852,
                    0.5297408103942871,
                    -0.5748386979103088,
                    -5.755568504333496
                ],
                [
                    -0.7817064523696899,
                    0.4226280152797699,
                    -0.45860719680786133,
                    -4.591801166534424
                ],
                [
                    -1.4901161193847656e-08,
                    0.7353639006614685,
                    0.677672266960144,
                    6.785189151763916
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]

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

def keypoint_3d_to_2d(keypoint3d: Keypoint) -> Keypoint:
    """
    Convert a 3D keypoint to a 2D keypoint representation.

    :param keypoint3d: The 3D keypoint to be converted.
    :type keypoint3d: Keypoint

    :return: The 2D keypoint representation of the input 3D keypoint.
    :rtype: Keypoint
    """
    location2d = bproc.camera.project_points(np.array([keypoint3d.location]))[0]
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

def write_to_json(keypoints, output_dir):
    with open(output_dir, "w") as f:
        serializable_keypoints = [k.to_serializable_dict() for k in keypoints]
        pretty = json.dumps(serializable_keypoints, indent=4)
        f.write(pretty)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', nargs='?', help="Path to the 3d model in .ply format")
    parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved")
    args = parser.parse_args()
        
    bproc.init()
    #Avoid this method!!
    with open('./src/dataset/transforms_train.json', 'r') as file:
        json_data = file.read()

    # read the camera positions file and convert into homogeneous camera-world transformation
    data = json.loads(json_data)
    trasformation_matrix_list = []

    ## Just one
    transform_matrix = data['frames'][0]['transform_matrix']
    trasformation_matrix_list.append(transform_matrix)

    ## All
    #for frame in data['frames']:
    #    transformation_matrix = frame['transform_matrix']
    #    trasformation_matrix_list.append(transformation_matrix)

    for matrix in trasformation_matrix_list:
        bproc.camera.add_camera_pose(matrix)

    scene = bproc.loader.load_blend(args.scene)
    bproc.camera.set_resolution(800, 800)

    keypoints3d = extract_3d_keypoints(scene)
    keypoints2d = [keypoint_3d_to_2d(keypoint) for keypoint in keypoints3d]
    write_to_json(keypoints2d, args.output_dir+"keypoints2d.json") #coco

    # collect all RGB paths
    new_coco_image_paths = []
    data = bproc.renderer.render()
    colors = data["colors"]
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    # for each rendered frame
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
        color_rgb = colors[frame - bpy.context.scene.frame_start]

        # Reverse channel order for opencv
        color_bgr = color_rgb.copy()
        color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]

        target_base_path = f'images/{00}{frame}.jpg' #TODO
        target_path = os.path.join(args.output_dir, target_base_path)
        print(target_path)
        cv2.imwrite(target_path, color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])



        new_coco_image_paths.append(target_base_path)
   

if __name__ == "__main__":
    main()