import os
import shutil
import random
import datetime
from pathlib import Path
import open3d as o3d
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains

def generate_random_number(lower_bound, upper_bound):
    # 生成一个在给定上下界之间的随机浮点数
    random_number = random.uniform(lower_bound, upper_bound)
    return random_number

def random_voxel_downsample_vectorized(points, voxel_size):
    """
    向量化的随机体素下采样实现
    
    参数:
    points: np.array, shape (N, D) - 输入点云
    voxel_size: float - 体素大小
    
    返回:
    np.array, shape (M, D) - 下采样后的点云及其索引
    """
    # 添加随机扰动到体素大小
    perturbed_voxel_size = voxel_size * (1 + np.random.uniform(-0.1, 0.1))
    
    # 向量化计算体素坐标
    random_offsets = np.random.uniform(-0.1, 0.1, size=(points.shape[0], 3))
    voxel_coords = np.floor(points / perturbed_voxel_size + random_offsets)
    
    # 使用字符串作为键以提高效率
    voxel_keys = [f"{x:.0f}_{y:.0f}_{z:.0f}" for x, y, z in voxel_coords]
    
    # 创建唯一体素到点的映射
    unique_voxels = {}
    for idx, key in enumerate(voxel_keys):
        if key not in unique_voxels or np.random.random() < 0.3:
            unique_voxels[key] = idx
    
    # 获取选中点的索引
    selected_indices = list(unique_voxels.values())
    
    return points[selected_indices], selected_indices

def distance_based_downsample(points, colors, reference_points, max_radius, min_keep_ratio=0.1, max_keep_ratio=0.9):
    """
    基于参考点距离的自适应降采样
    
    参数:
    points: np.array, shape (N, 3) - 输入点云
    colors: np.array, shape (N, 3) - 点云颜色
    reference_points: list of [x, y, z] - 参考点列表
    max_radius: float - 最大影响半径
    min_keep_ratio: float - 最远处的保留比例
    max_keep_ratio: float - 最近处的保留比例
    
    返回:
    tuple (np.array, np.array) - 降采样后的点云和对应的颜色
    """
    if len(points) == 0:
        return points, colors
    
    # 计算每个点到所有参考点的最小距离
    min_distances = np.inf * np.ones(len(points))
    for ref_point in reference_points:
        ref_point = np.array(ref_point)
        distances = np.linalg.norm(points - ref_point, axis=1)
        min_distances = np.minimum(min_distances, distances)
    
    # 将距离归一化到[0, 1]范围
    normalized_distances = np.clip(min_distances / max_radius, 0, 1)
    
    # 计算每个点的保留概率（距离越远，保留概率越小）
    keep_ratios = max_keep_ratio - (max_keep_ratio - min_keep_ratio) * normalized_distances
    
    # 随机采样
    random_values = np.random.random(len(points))
    mask = random_values < keep_ratios
    
    return points[mask], colors[mask]

def crop_point_cloud_efficient_vectorized(points, colors, crop_polygon):
    """使用shapely的向量化操作进行高效裁剪"""
    # 创建边界框以加速处理
    min_x, min_y = np.min(crop_polygon, axis=0)
    max_x, max_y = np.max(crop_polygon, axis=0)
    
    # 快速筛选边界框内的点
    in_bbox = (points[:, 0] >= min_x) & (points[:, 0] <= max_x) & \
              (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    
    # 对边界框内的点进行精确多边形判断
    polygon = Polygon(crop_polygon)
    points_in_bbox = points[in_bbox]
    mask = contains(polygon, points_in_bbox[:, 0], points_in_bbox[:, 1])
    
    # 获取最终的掩码
    final_mask = np.zeros(len(points), dtype=bool)
    final_mask[in_bbox] = mask
    
    return points[final_mask], colors[final_mask]

def process_single_camera(camera_file, camera_voxel_size, crop_polygon):
    """处理单个相机的点云数据"""
    pcd = o3d.io.read_point_cloud(str(camera_file))
    if not pcd.has_points():
        return None, None
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 先进行裁剪
    points, colors = crop_point_cloud_efficient_vectorized(points, colors, crop_polygon)
    
    # 再进行降采样
    if camera_voxel_size > 0 and len(points) > 0:
        downsampled_points, indices = random_voxel_downsample_vectorized(points, camera_voxel_size)
        downsampled_colors = colors[indices]
        return downsampled_points, downsampled_colors
    
    return points, colors

def merge_ply_files(scene_folder, selected_cameras, output_ply_dir, output_label_dir, 
                   crop_polygon, camera_voxel_size, merged_voxel_size, 
                   reference_points, max_radius, min_keep_ratio=0.1, max_keep_ratio=0.9):
    """
    优化后的PLY文件合并函数，包含三阶段降采样：
    1. 单相机降采样
    2. 合并后的体素降采样
    3. 基于参考点距离的自适应降采样
    """
    # 创建输出目录
    Path(output_ply_dir).mkdir(parents=True, exist_ok=True)
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)
    
    scene_path = Path(scene_folder)
    if not scene_path.exists():
        print(f"错误：目录 {scene_folder} 不存在")
        return
    
    # 处理每个时间戳目录
    for timestamp_dir in scene_path.glob("*-*-*-*-*-*"):
        timestamp_points = []
        timestamp_colors = []
        
        # 生成输出文件名
        current_time = datetime.datetime.now()
        random_num = random.randint(1000, 9999)
        output_filename = f"{current_time.strftime('%H%M%S%f')[:-3]}_{random_num}"
        
        # 处理每个相机
        for camera_num in selected_cameras:
            camera_file = timestamp_dir / f"Camera{camera_num}.ply"
            if not camera_file.exists():
                print(f"警告：文件 {camera_file} 不存在")
                continue
                
            # 处理单个相机数据
            points, colors = process_single_camera(camera_file, camera_voxel_size, crop_polygon)
            if points is not None and len(points) > 0:
                timestamp_points.append(points)
                timestamp_colors.append(colors)
        
        if timestamp_points:
            # 合并点云数据
            merged_points = np.vstack(timestamp_points)
            merged_colors = np.vstack(timestamp_colors)
            
            # 第一阶段：体素降采样
            if merged_voxel_size > 0:
                merged_points, indices = random_voxel_downsample_vectorized(merged_points, merged_voxel_size)
                merged_colors = merged_colors[indices]
            
            # 第二阶段：基于距离的自适应降采样
            merged_points, merged_colors = distance_based_downsample(
                merged_points, 
                merged_colors,
                reference_points,
                max_radius,
                min_keep_ratio,
                max_keep_ratio
            )
            
            # 创建和保存点云
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(merged_points)
            filtered_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            
            # 保存结果
            output_ply_path = Path(output_ply_dir) / f"{output_filename}.ply"
            o3d.io.write_point_cloud(str(output_ply_path), filtered_pcd)
            print(f"已创建降采样后的合并文件：{output_ply_path}")
            
            # 复制标签文件
            target_info_file = timestamp_dir / "Camera1-TargetInfo.txt"
            if target_info_file.exists():
                output_label_path = Path(output_label_dir) / f"{output_filename}.txt"
                shutil.copy2(target_info_file, output_label_path)
                print(f"已复制标签文件：{output_label_path}")
            else:
                print(f"警告：标签文件 {target_info_file} 不存在")

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"目录 '{directory_path}' 已创建。")
    else:
        print(f"目录 '{directory_path}' 已存在。")

if __name__ == "__main__":
    # 参数设置
    #   scene_folder: 场景文件夹
    #   camera_voxel_size: 相机体素大小
    #   merged_voxel_size: 合并后体素大小
    #   reference_points: 参考点
    #   max_radius: 最大半径
    #   min_keep_ratio: 最小保留比例
    #   max_keep_ratio: 最大保留比例
    scene_name_list = ['Scene1_1', 'Scene1_2' ]
    for scene_name in scene_name_list:
        scene_folder = scene_name
        
        # 选择相机组合
        list_1 = [[1, 6, 7], [2, 5, 8], [3, 4, 9], [1, 5, 9], [3, 5, 7], [2, 6, 8], [3, 6, 8], [1, 4, 8], [3, 6, 9], [1, 4, 7], [1, 3, 5, 7, 9], [1, 3, 4, 6, 7], [1, 2, 4, 5, 8], [1, 4, 6, 7, 8]]     
        for sub_list_1 in list_1:
            selected_cameras = sub_list_1    # 1 2 3 4 5 6 7 8   
            print('sub_list_1 is', sub_list_1) 
            file_name_aa = ''
            for element_list in sub_list_1: 
                file_name_aa = str(file_name_aa + str(element_list))
            output_ply_dir = scene_folder + "_output/" + file_name_aa + "/ply/" 
            output_label_dir = scene_folder + "_output/" + file_name_aa + "/label/"

            A = random.randint(1, 4)
            B = random.randint(1, 4)
            # //////////////////////////////////////////////////////////////////////////
            if A == 1:
                camera_voxel_size = generate_random_number(0.080, 0.085) 
                merged_voxel_size = generate_random_number(0.080, 0.085) 
            elif A== 2:
                camera_voxel_size = generate_random_number(0.075, 0.08) 
                merged_voxel_size = generate_random_number(0.075, 0.08) 

            elif A== 3:
                camera_voxel_size = generate_random_number(0.07, 0.075) 
                merged_voxel_size = generate_random_number(0.07, 0.075) 

            elif A == 4:
                camera_voxel_size = generate_random_number(0.065, 0.07) 
                merged_voxel_size = generate_random_number(0.065, 0.07) 


            
            if B == 1:
                min_keep_ratio = generate_random_number(0.3, 0.35)   # 最远处的保留比例
                max_keep_ratio = generate_random_number(0.75, 0.8)    # 最近处的保留比例
            elif B == 2:
                min_keep_ratio = generate_random_number(0.35, 0.4)   # 最远处的保留比例
                max_keep_ratio = generate_random_number(0.8, 0.85)    # 最近处的保留比例
            elif B == 3:
                min_keep_ratio = generate_random_number(0.4, 0.45)   # 最远处的保留比例
                max_keep_ratio = generate_random_number(0.85, 0.9)    # 最近处的保留比例
            elif B == 4:
                min_keep_ratio = generate_random_number(0.45, 0.5)   # 最远处的保留比例
                max_keep_ratio = generate_random_number(0.9, 1)    # 最近处的保留比例



            if scene_name == 'Scene1_1' or scene_name == 'Scene1_2' or scene_name == 'Scene1_3' or scene_name == 'Scene1_4':
                x1 = generate_random_number(25, 27)
                y1 = generate_random_number(5, 7)
                x2 = generate_random_number(-22, -20)
                y2 = generate_random_number(45, 48)

                p1 = generate_random_number(5.5, 7.5)   # 最远处的保留比例
                q1 = generate_random_number(22, 28)    # 最近处的保留比例
                v1 = generate_random_number(22, 28)   # 最远处的保留比例
                p2 = generate_random_number(15, 20)    # 最近处的保留比例
                q2 = generate_random_number(28, 36)   # 最远处的保留比例
                v2 = generate_random_number(8, 20)    # 最近处的保留比例
            # ------------------------------------------

            elif scene_name == 'Scene2_1' or scene_name == 'Scene2_2':
                x1 = generate_random_number(16, 18)
                y1 = generate_random_number(4, 6)
                x2 = generate_random_number(-16, -14)
                y2 = generate_random_number(45, 48)

                p1 = generate_random_number(5.5, 7.5)   # 最远处的保留比例
                q1 = generate_random_number(22, 28)    # 最近处的保留比例
                v1 = generate_random_number(22, 28)   # 最远处的保留比例
                p2 = generate_random_number(10, 12)    # 最近处的保留比例
                q2 = generate_random_number(34, 38)   # 最远处的保留比例
                v2 = generate_random_number(8, 20)    # 最近处的保留比例
            
             # ------------------------------------------

            elif scene_name == 'Scene3_1' or scene_name == 'Scene3_2':
                x1 = generate_random_number(23, 25)
                y1 = generate_random_number(4, 6)
                x2 = generate_random_number(-23, -20)
                y2 = generate_random_number(45, 48)

                p1 = generate_random_number(5.5, 7.5)   # 最远处的保留比例
                q1 = generate_random_number(22, 28)    # 最近处的保留比例
                v1 = generate_random_number(22, 28)   # 最远处的保留比例
                p2 = generate_random_number(10, 12)    # 最近处的保留比例
                q2 = generate_random_number(34, 38)   # 最远处的保留比例
                v2 = generate_random_number(8, 20)    # 最近处的保留比例

            # 裁剪区域设置
            crop_polygon = [
                [x1, y1],
                [x1, y2],
                [x2, y2],
                [x2, y1]
            ]
            
            # 新增：基于距离降采样的参数
            reference_points = [
                [p1, q1, v1],  # 第一个参考点
                [p2, q2, v2]   # 第二个参考点
            ]

            max_radius = generate_random_number(15, 25)        # 最大影响半径

            merge_ply_files(
                scene_folder,
                selected_cameras,
                output_ply_dir,
                output_label_dir,
                crop_polygon,
                camera_voxel_size,
                merged_voxel_size,
                reference_points,
                max_radius,
                min_keep_ratio,
                max_keep_ratio
            )
