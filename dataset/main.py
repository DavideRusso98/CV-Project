

""" 
to annotatate: blenderproc run main.py camera_positions tesla.blend output
(Camera position non serve a un cazzo, perchè usiamo la matrix, basta mettere un file a caso)
to visualize: blenderproc vis coco -i 25 -c coco_annotations.json -b blender/coco_data/
Reference:
https://github.com/DLR-RM/BlenderProc/tree/main/examples/advanced/coco_annotations
"""
import blenderproc as bproc
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('camera', nargs='?', default="examples/resources/camera_positions", help="Path to the camera file")
parser.add_argument('scene', nargs='?', default="examples/advanced/coco_annotations/scene.blend", help="Path to the scene.blend file")
parser.add_argument('output_dir', nargs='?', default="examples/advanced/coco_annotations/output", help="Path to where the final files will be saved ")
args = parser.parse_args()

bproc.init()

# load the objects into the scene
objs = bproc.loader.load_blend(args.scene)

# Set some category ids for loaded objects
for j, obj in enumerate(objs):
    obj.set_cp("category_id", j + 1)

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

# define the camera intrinsics
bproc.camera.set_resolution(800, 800)

#Avoid this method!!
with open('transforms_train.json', 'r') as file:
    json_data = file.read()
# read the camera positions file and convert into homogeneous camera-world transformation
data = json.loads(json_data)
#transform_matrix = data['frames'][0]['transform_matrix']
trasformation_matrix_list = []
for frame in data['frames']:
    transformation_matrix = frame['transform_matrix']
    trasformation_matrix_list.append(transformation_matrix)

for matrix in trasformation_matrix_list:
    bproc.camera.add_camera_pose(matrix)


""" with open(args.camera, "r") as f:
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        position, euler_rotation = line[:3], line[3:6]
        #matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        matrix_world = transform_matrix# bproc.math.build_transformation_mat(position, euler_rotation)
        print("####\n",matrix_world)
        bproc.camera.add_camera_pose(matrix_world) """

# activate normal rendering
bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# render the whole pipeline
data = bproc.renderer.render()

# Write data to coco file
bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG")
