def computeCov3D(scale, mod, rot):
    # 创建缩放矩阵
    S = np.array(
        [[scale[0] * mod, 0, 0], [0, scale[1] * mod, 0], [0, 0, scale[2] * mod]]
    )

    # 归一化四元数以获得有效的旋转矩阵
    # 我们使用旋转矩阵
    R = rot

    # 计算3D世界的协方差矩阵Sigma
    M = np.dot(R, S)
    cov3D = np.dot(M, M.T)

    return cov3D
