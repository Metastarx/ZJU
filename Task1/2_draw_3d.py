import os
import cv2
import numpy as np
import time
import open3d as o3d
from data.kitti_Dataset import *

# 根据偏航角计算旋转矩阵（逆时针旋转）
def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R

# 绘制3D框架
def draw_3dframeworks(vis, points):
    position = points
    points_box = np.transpose(position)

    # 定义3D框架的边
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7], [0, 5], [1, 4]])

    # 每条边的颜色
    colors = np.array([[1., 0., 0.] for j in range(len(lines_box))])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_box)
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    render_option = vis.get_render_option()
    render_option.line_width = 5.0
    render_option.background_color = np.asarray([1, 1, 1])
    render_option.point_size = 4

    vis.add_geometry(line_set)
    vis.update_geometry(line_set)
    vis.update_renderer()

if __name__ == "__main__":
    dir_path = "data/kitti/"
    index = 10  # 图片的标号
    split = "training"

    # 创建KITTI数据集对象
    dataset = Kitti_Dataset(dir_path, split=split)

    # 创建Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=771, height=867)

    # 获取标签、RGB图像、校准信息和点云数据
    obj = dataset.get_labels(index)
    img3_d = dataset.get_rgb(index)
    calib1 = dataset.get_calib(index)
    pc = dataset.get_pcs(index)

    # 创建点云对象并设置颜色
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.paint_uniform_color([0, 121/255, 89/255])
    vis.add_geometry(point_cloud)

    render_option = vis.get_render_option()
    render_option.line_width = 4

    # 遍历每个物体
    for obj_index in range(len(obj)):
        if obj[obj_index].name == "Car" or obj[obj_index].name == "Pedestrian" or obj[obj_index].name == "Cyclist":
            R = rot_y(obj[obj_index].rotation_y)
            h, w, l = obj[obj_index].dimensions[0], obj[obj_index].dimensions[1], obj[obj_index].dimensions[2]

            # 定义物体的角点坐标
            x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y = [0, 0, 0, 0, -h, -h, -h, -h]
            z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

            # 经过旋转后的角点坐标
            corner_3d = np.vstack([x, y, z])
            corner_3d = np.dot(R, corner_3d)

            # 将物体移动到相机坐标系原点
            corner_3d[0, :] += obj[obj_index].location[0]
            corner_3d[1, :] += obj[obj_index].location[1]
            corner_3d[2, :] += obj[obj_index].location[2]
            corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
            corner_3d[-1][-1] = 1

            # 计算逆变换矩阵
            inv_Tr = np.zeros_like(calib1.Tr_velo_to_cam)
            inv_Tr[0:3, 0:3] = np.transpose(calib1.Tr_velo_to_cam[0:3, 0:3])
            inv_Tr[0:3, 3] = np.dot(-np.transpose(calib1.Tr_velo_to_cam[0:3, 0:3]), calib1.Tr_velo_to_cam[0:3, 3])

            # 转换到相机坐标系下
            Y = np.dot(inv_Tr, corner_3d)

            # 绘制物体的3D框架
            draw_3dframeworks(vis, Y)

    # 运行可视化
    vis.run()
