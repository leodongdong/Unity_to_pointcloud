import numpy as np
import os
import open3d as o3d


def ply_to_npy(ply_path, npy_path):
    """
        将pcd文件转换成npy文件
        pcd_path: pcd格式点云目录
        npy_path: npy格式点云输出目录
    """
    pcd_list = os.listdir(ply_path)
    
    for pcd_name in pcd_list:
        lidar = []
        ply_file = os.path.join(ply_path, pcd_name)
        ply = o3d.io.read_point_cloud(ply_file)
        points = np.array(ply.points)
        
        # print('points is', points)
        for linestr in points:
            linestr_convert = list(map(float, linestr))
            linestr_convert.append(0)
            lidar.append(linestr_convert)

        lidar = np.array(lidar).astype(np.float32)
        # print('lidar is', lidar)
        np.save(os.path.join(npy_path, pcd_name[:-4]+".npy"), lidar)

# 使用示例
input_folder = "Ply"
output_folder = "Convert_npy"
ply_to_npy(input_folder, output_folder)
