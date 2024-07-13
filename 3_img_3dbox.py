import cv2
import numpy as np
from data.kitti_Dataset import *

# 根据偏航角计算旋转矩阵（逆时针旋转）
def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R

if __name__ == "__main__":
    # 数据集文件夹路径和训练集划分
    dir_path = "data/kitti/"
    split = "training"
    dataset = Kitti_Dataset(dir_path, split=split)

    # 初始帧索引和最大帧数
    k = 10
    max_num = 100

    while True:
        # 获取RGB图像和标定数据
        img3_d = dataset.get_rgb(k)
        calib = dataset.get_calib(k)
        obj = dataset.get_labels(k)

        # 遍历每个对象的标签
        for num in range(len(obj)):
            # 只处理"Car", "Pedestrian", "Cyclist"类型的对象
            if obj[num].name == "Car" or obj[num].name == "Pedestrian" or obj[num].name == "Cyclist":
                # 计算旋转矩阵R
                R = rot_y(obj[num].rotation_y)
                h, w, l = obj[num].dimensions[0], obj[num].dimensions[1], obj[num].dimensions[2]

                # 计算物体在其坐标系下的8个顶点坐标
                x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                y = [0, 0, 0, 0, -h, -h, -h, -h]
                z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
                corner_3d = np.vstack([x, y, z])
                corner_3d = np.dot(R, corner_3d)

                # 将物体坐标转换到相机坐标系下
                corner_3d[0, :] += obj[num].location[0]
                corner_3d[1, :] += obj[num].location[1]
                corner_3d[2, :] += obj[num].location[2]

                # 增加齐次坐标，并将3D坐标转换为2D像素坐标
                corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
                corner_2d = np.dot(calib.P2, corner_3d)
                corner_2d[0, :] /= corner_2d[2, :]
                corner_2d[1, :] /= corner_2d[2, :]
                corner_2d = np.array(corner_2d, dtype=np.int)

                # 绘制3D边界框
                color = [0, 255, 0]
                thickness = 2
                for corner_i in range(0, 4):
                    i, j = corner_i, (corner_i + 1) % 4
                    cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)

                    i, j = corner_i + 4, (corner_i + 1) % 4 + 4
                    cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)

                    i, j = corner_i, corner_i + 4
                    cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)

                cv2.line(img3_d, (corner_2d[0, 0], corner_2d[1, 0]), (corner_2d[0, 5], corner_2d[1, 5]), color, thickness)
                cv2.line(img3_d, (corner_2d[0, 1], corner_2d[1, 1]), (corner_2d[0, 4], corner_2d[1, 4]), color, thickness)

        # 显示带有边界框的图像，并处理按键
        cv2.imshow("{}".format(k), img3_d)
        cv2.moveWindow("{}".format(k), 300, 50)
        key = cv2.waitKey(100) & 0xFF

        if key == ord('d'):
            k += 1
            cv2.destroyAllWindows()

        if key == ord('a'):
            k -= 1

        if key == ord('q'):
            break

        if k >= max_num:
            k = max_num - 1

        if k < 0:
            k = 0
