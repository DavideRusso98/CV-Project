import blenderproc as bproc
import json
import numpy as np
import argparse

##transformation matrix from camera to world coordinates
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

### Not used
def camera_manipulation_example():
    # Define the camera parameters
    cam_location = [2, -2, 1]
    cam_rotation = bproc.camera.rotation_from_euler_angles([1.1, 0, 0.8])
    cam = bproc.camera.Camera(location=cam_location, rotation_quaternion=cam_rotation)
    bproc.camera.set_intrinsics_from_blender_params(cam)

def load_keypoints_from_file():
    pass

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('scene', nargs='?', help="Path to the 3d model in .ply format")
    parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved")
    parser.add_argument('annotations', nargs='?', help="Path to the annotation file")
    args = parser.parse_args()

    # Initialize BlenderProc
    bproc.init()

    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    # Load the 3D model
    ###obj = bproc.loader.load_obj(filepath="path_to_your_model.ply")[0]
    objs = bproc.loader.load_blend(args.scene)

    ### TODO: load keypoints from a file
    # Load keypoints from annotations.json
    #with open(args.annotations, "r") as f:
    #    keypoints_3d = json.load(f)
    keypoint3d = np.array([0.14804799854755402, -0.032051329126306205, -0.19088649899083138]) ##copied alla brutta

    ### Define a camera
    bproc.camera.add_camera_pose(transform_matrix)
    bproc.camera.set_resolution(800, 800)
    bproc.renderer.enable_depth_output(False) ##boh
    bproc.renderer.enable_normals_output(False) ##boh

    ### Project a 3d keypoint into the 2d rendering
    keypoint2d = bproc.camera.project_points(keypoint3d)

    # Render the scene
    data = bproc.renderer.render()

    # Save the rendered image
    bproc.writer.write_rendered_data(data, args.output)

    # Save the 2D keypoints annotations
    with open(args.output + "/keypoints_2d_annotations.json", "w") as f:
        json.dump(keypoint2d, f)

if __name__ == "__main__":
    main()