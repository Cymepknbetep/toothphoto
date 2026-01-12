# 归一化牙齿
import trimesh
from mesh_to_sdf import mesh_to_voxels
import skimage
import imageio
import numpy as np
import pyrender
import os
from scipy.ndimage import gaussian_filter  # 用于 Voxel 平滑
import trimesh.smoothing                  # 用于 Mesh 平滑

# 加载牙齿的 OBJ 文件
gen_again = False
mesh = trimesh.load('../data/mesh/teeth_down.stl')
# 检查 mesh 是否有效
print(f"Is the mesh watertight? {mesh.is_watertight}")
print(f"Number of vertices: {len(mesh.vertices)}")
print(f"Number of faces: {len(mesh.faces)}")
# 可选：修复 mesh（如果需要）
if not mesh.is_watertight:
    print("Attempted to repair mesh.")
    # # 将原始 mesh 拆分为独立的连通分量（每颗牙齿一个）
    # teeth_list = mesh.split(only_watertight=False)
    # print(f"Found {len(teeth_list)} individual teeth.")

    # processed_teeth = []
    # for i, tooth in enumerate(teeth_list):
    #     # 过滤掉极其微小的杂质碎块
    #     if tooth.area < 1e-5: # 根据你的缩放比例调整
    #         continue

    #     # 强力修复每一颗牙齿
    #     tooth.remove_infinite_values()
    #     tooth.fill_holes()  # 封死底部开口
    #     tooth.fix_normals()
    #     processed_teeth.append(tooth)

    # # 重新组合
    # mesh = trimesh.util.concatenate(processed_teeth)
    mesh.fill_holes()

# 缩放
max_scale = 0.05 #成年人牙齿约间隔0.05m
# 定义腐蚀量
erosion_amount = 0.5  # 调整腐蚀程度，单位与SDF值一致

if gen_again or not os.path.exists("../temp/voxels.npy"):
    # 1. 提高分辨率：64 对整排牙齿来说太低了，牙缝会粘连。建议 128 或更高。
    # 2. 更改判定方法：sign_method='depth' 对非闭合或多物体模型更鲁棒。
    # 3. surface_point_method='sample' 在处理独立个体时有时比 scan 更稳。

    voxels = mesh_to_voxels(mesh, 
                            voxel_resolution=256, 
                            pad=True, 
                            sign_method='depth', 
                            surface_point_method='scan')
    np.save("../temp/voxels.npy",voxels)
voxels = np.load("../temp/voxels.npy")

# --- 2. Voxel 级别平滑 (关键：消除方块感的源头) ---
# sigma 决定平滑程度。0.5~1.0 之间效果最好。
# 它会让 SDF 的数值过渡更连续，从而让 Marching Cubes 产生更平滑的斜面。
print("Applying Gaussian filter to voxels...")
voxels_smoothed = gaussian_filter(voxels, sigma=0.8)
voxels_modified_smoothed = gaussian_filter(voxels + erosion_amount*max_scale, sigma=0.8)




vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels_modified_smoothed, level=0)
eroded_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels_smoothed, level=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)





# 步骤 1: 居中化
# 计算 mesh 的几何中心（质心）
centroid = mesh.centroid

# 将 mesh 移动到原点
mesh.apply_translation(-centroid)
eroded_mesh.apply_translation(-centroid)
# 步骤 2: 归一化到0.1单位立方体
# 计算 mesh 的边界框范围
bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
extent = mesh.extents  # [x_range, y_range, z_range]
# 找到最大范围
max_extent = np.max(extent)
# 缩放因子：将最大范围缩放到 0.01
scale_factor = max_scale / max_extent


# 应用缩放
mesh.apply_scale(scale_factor)
eroded_mesh.apply_scale(scale_factor)

# ==========================================
# 新增部分：创建双层牙齿 (Double Layer)
# ==========================================

print("Creating upper teeth layer...")

# 1. 定义参数
# 垂直位移量：决定上下牙齿之间的间隙大小
# 因为当前模型总高度约 0.1，向上下各移动 0.06 大约能留出一点空隙
shift_distance = 0.01

# 2. 准备下排牙齿（原始）
# 复制一份作为最终的下排
lower_mesh_final = mesh.copy()
lower_eroded_final = eroded_mesh.copy()

# 向下移动 (-Z 方向)
translation_down = trimesh.transformations.translation_matrix([0, 0, -shift_distance])
lower_mesh_final.apply_transform(translation_down)
lower_eroded_final.apply_transform(translation_down)

# 3. 创建上排牙齿（通过镜像和移动）
upper_mesh_final = mesh.copy()
upper_eroded_final = eroded_mesh.copy()

# --- 3a. 镜像反射 (Reflection) ---
# 创建一个沿 Z 轴反射的矩阵 (Z 坐标变为负数)
reflection_matrix = np.eye(4)
reflection_matrix[2, 2] = -1
# 应用反射
upper_mesh_final.apply_transform(reflection_matrix)
upper_eroded_final.apply_transform(reflection_matrix)

# [重要] 修复法线：反射变换会导致法线指向内部，必须修复
upper_mesh_final.fix_normals()
upper_eroded_final.fix_normals()

# --- 3b. 向上移动 (+Z 方向) ---
translation_up = trimesh.transformations.translation_matrix([0, 0, shift_distance])
upper_mesh_final.apply_transform(translation_up)
upper_eroded_final.apply_transform(translation_up)

# 4. 合并上下排
print("Combining upper and lower layers...")
combined_mesh = trimesh.util.concatenate([lower_mesh_final, upper_mesh_final])
combined_eroded_mesh = trimesh.util.concatenate([lower_eroded_final, upper_eroded_final])


# ==========================================
# 新增：绕 +X 轴顺时针旋转 90 度
# ==========================================
# 顺时针 90 度即 -pi/2 弧度
rotation_matrix = trimesh.transformations.rotation_matrix(
    angle=np.pi/2, 
    direction=[1, 0, 0], 
    point=[0, 0, 0]
)
combined_mesh.apply_transform(rotation_matrix)
combined_eroded_mesh.apply_transform(rotation_matrix)
# ==========================================


# 5. 设置材质（可选，为了 MeshLab 查看方便）
combined_mesh.visual.material = trimesh.visual.material.SimpleMaterial(
    diffuse=[200, 200, 200, 255]
)
combined_eroded_mesh.visual.material = trimesh.visual.material.SimpleMaterial(
    diffuse=[255, 100, 100, 255] # 腐蚀版用红色区分
)

# 6. 导出最终结果
output_dir = "../data/mesh/"
os.makedirs(output_dir, exist_ok=True) # 确保目录存在

combined_mesh_path = os.path.join(output_dir, "teeth_double_layer.obj")
combined_eroded_path = os.path.join(output_dir, "teeth_double_layer_eroded.obj")

print(f"Exporting combined mesh to {combined_mesh_path}...")
combined_mesh.export(combined_mesh_path)

print(f"Exporting combined eroded mesh to {combined_eroded_path}...")
combined_eroded_mesh.export(combined_eroded_path)

print("Done!")
# 打印一下最终信息看看
print(f"Final combined mesh vertices: {len(combined_mesh.vertices)}")
print(f"Final combined bounds Z-range: {combined_mesh.bounds[:, 2]}")