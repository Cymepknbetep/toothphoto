# 归一化人头
import trimesh
from mesh_to_sdf import mesh_to_voxels
import imageio
import numpy as np
import pyrender

material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    roughnessFactor=1.0,
    baseColorFactor=(0.5, 0.5, 0.5, 1.0)  # 纯白色
)

mesh = trimesh.load_mesh('Head.obj',material=material)

# 检查 mesh 是否有效
print(f"Is the mesh watertight? {mesh.is_watertight}")
print(f"Number of vertices: {len(mesh.vertices)}")
print(f"Number of faces: {len(mesh.faces)}")

# 可选：修复 mesh（如果需要）
if not mesh.is_watertight:
    mesh.fill_holes()  # 尝试修复非封闭的 mesh
    print("Attempted to repair mesh.")


# 步骤 1: 居中化
# 计算 mesh 的几何中心（质心）
centroid = mesh.centroid

# 将 mesh 移动到原点
mesh.apply_translation(-centroid)
# 步骤 2: 归一化到单位立方体
# 计算 mesh 的边界框范围
bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
extent = mesh.extents  # [x_range, y_range, z_range]
# 找到最大范围
max_extent = np.max(extent)
# 缩放因子：将最大范围缩放到 0.25
scale_factor = 0.35 / max_extent
# 应用缩放
mesh.apply_scale(scale_factor)

mesh.visual.material = trimesh.visual.material.SimpleMaterial(
    diffuse=[255, 255, 255, 255]  # 白色材质
)

mesh.export('head_mesh.obj')