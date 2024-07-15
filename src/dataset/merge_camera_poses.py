import json


def merge_camera_poses(data_high, data_low):
    out = []
    transformation_matrix_list_1 = [frame['transform_matrix'] for frame in data_high['frames']]
    transformation_matrix_list_2 = [frame['transform_matrix'] for frame in data_low['frames']]
    for i in range(len(transformation_matrix_list_1)):
        if i % 2 == 0:
            out.append(transformation_matrix_list_1[i])
        else:
            out.append(transformation_matrix_list_2[i])
    return out


def main():
    with open('./src/dataset/transforms_train_high.json', 'r') as file:
        json_data = file.read()
    data_high = json.loads(json_data)

    with open('./src/dataset/transforms_train_low.json', 'r') as file:
        json_data = file.read()
    data_low = json.loads(json_data)

    data = merge_camera_poses(data_high, data_low)
    with open('./src/dataset/camera_poses.json', 'w') as file:
        json.dump(data, file)

if __name__ == '__main__':
    main()
