import os
import argparse
import numpy as np
from map_reader import MapReader
import matplotlib.pyplot as plt

def precompute_ray_casting(occupancy_map, max_range=1000, step_size=2, num_angles=120):

    lookup_table = np.full((800, 800, num_angles), 0, dtype=np.float32)

    angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)

    for x in range(800):
        print(f"x = {x}...")
        for y in range(800):
            # print(f"x = {x}, y = {y}...")
            if occupancy_map[y][x] != 0:
                continue

            for theta_idx, theta in enumerate(angles):

                # Ray Casting
                for r in range(0, max_range, step_size):
                    x_new = x + r * np.cos(theta)
                    y_new = y + r * np.sin(theta)

                    map_x = int(round(x_new))
                    map_y = int(round(y_new))

                    if map_x < 0 or map_x >= 800 or map_y < 0 or map_y >= 800:
                        lookup_table[x, y, theta_idx] = r*10
                        break

                    if occupancy_map[map_y, map_x] > 0.35:
                        lookup_table[x, y, theta_idx] = r*10
                        break

    return lookup_table

def visualize_lookup_table_on_map(occupancy_map, lookup_table, selected_points, num_angles=120):
    """
    在地图上绘制多个点的 Ray Casting 结果。

    参数：
    - occupancy_map: 800x800 占据网格地图
    - lookup_table: 预计算的 Ray Casting 查找表 (800, 800, num_angles)
    - selected_points: 选取的 3 个点 [(x1, y1), (x2, y2), (x3, y3)]
    - num_angles: 角度数量 (例如 120 表示每 3° 计算一次)
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制地图（背景）
    ax.imshow(occupancy_map, cmap="gray", origin="lower")

    # 生成角度索引对应的角度值 (-π 到 π)
    angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)

    # 变量：存储点的图例信息，防止重复
    plotted_labels = set()

    # 遍历选定的点
    for (x, y) in selected_points:
        # 在地图上标记起点
        label = f"Start ({x},{y})"
        if label not in plotted_labels:  # 确保每个点的标签只出现一次
            ax.scatter(x, y, color="red", s=100, marker="x", label=label)
            plotted_labels.add(label)

        # 遍历所有角度
        for theta_idx in range(num_angles):
            theta = angles[theta_idx]  # 获取当前角度
            distance = lookup_table[x, y, theta_idx] / 10.0  # 获取该点该角度的测距值
            
            # 计算激光束终点坐标
            x_end = int(x + distance * np.cos(theta))
            y_end = int(y + distance * np.sin(theta))

            # 仅绘制合理范围内的测距结果
            if 0 <= x_end < 800 and 0 <= y_end < 800:
                ax.plot([x, x_end], [y, y_end], color="yellow", alpha=0.5, linewidth=0.7)  # 画射线

    ax.set_title("Ray Casting Lookup Table Overlay on Map (Selected Points)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='data/map/wean.dat')
    parser.add_argument('--calculate', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()

    if args.calculate:
        print("Precalculating...")
        lookup_table = precompute_ray_casting(occupancy_map)
        save_path = "data/lookup_table2.npy"
        np.save(save_path, lookup_table)
        print("Lookup table saved!")

    if args.visualize:
        lookup = np.load('data/lookup_table.npy')
        selected_points = [(50, 50), (700, 700), (420, 600), (400, 400), (460, 220), (580, 150)]
        visualize_lookup_table_on_map(occupancy_map, lookup, selected_points)
