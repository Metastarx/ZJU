frame = create_canvas(700, 700)  # 创建一个画布
angle = 0  # 初始角度设为0
eye = [0, 0, 5]  # 观察者的位置
pts = [[2, 0, -2], [0, 2, -2], [-2, 0, -2]]  # 三个3D点的坐标
viewport = get_viewport_matrix(700, 700)  # 获取视口变换矩阵，将裁剪坐标系映射到屏幕坐标系

# 获取模型视图投影（MVP）矩阵
mvp = get_model_matrix(angle)  # 获取模型变换矩阵
mvp = np.dot(get_view_matrix(eye), mvp)  # 将视图变换矩阵应用到MVP矩阵
mvp = np.dot(get_proj_matrix(45, 1, 0.1, 50), mvp)  # 将投影变换矩阵应用到MVP矩阵（4x4）

# 遍历每个点
pts_2d = []
for p in pts:
    p = np.array(p + [1])  # 将3维点扩展为4维（齐次坐标）
    p = np.dot(mvp, p)  # 应用MVP矩阵
    p /= p[3]  # 进行透视除法，得到裁剪空间坐标

    # 视口变换
    p = np.dot(viewport, p)[:2]  # 应用视口变换矩阵，得到屏幕坐标系下的二维坐标
    pts_2d.append([int(p[0]), int(p[1])])  # 将坐标转换为整数

vis = 1  # 可视化开关
if vis:
    # 可视化3D效果
    fig = plt.figure()
    pts = np.array(pts)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    
    ax = Axes3D(fig)
    ax.scatter(x, y, z, s=80, marker="^", c="g")  # 绘制散点图
    ax.scatter([eye[0]], [eye[1]], [eye[2]], s=180, marker=7, c="r")  # 绘制观察者位置
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, alpha=0.5)  # 绘制三角网格表面
    plt.show()  # 显示图形
    
    # 可视化2D效果
    c = (255, 255, 255)  # 线条颜色
    for i in range(3):
        for j in range(i + 1, 3):
            cv2.line(frame, pts_2d[i], pts_2d[j], c, 2)  # 绘制3D点在屏幕上的投影线段
    cv2.imshow("screen", frame)  # 在窗口中显示结果
    cv2.waitKey(0)  # 等待按键事件
