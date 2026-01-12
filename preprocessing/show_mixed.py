# 显示头和牙齿的效果
# 简单的展示
import trimesh
import pyrender
import numpy as np
import OpenGL.GL as gl

def render(name,intensity):
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.5,
        baseColorFactor=(0.5, 0.5, 0.5, 1),  # 纯白色
    )
    pymesh = pyrender.Mesh.from_trimesh(trimesh.load_mesh(name),material=material)
    node = pyrender.Node(mesh=pymesh)
    scene = pyrender.Scene(bg_color=[255,255,255])
    light=pyrender.DirectionalLight(intensity=intensity)
    #light.direction = np.array([0,0,-1])
    light = pyrender.Node(light=light)
    cam = pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=np.pi/3,aspectRatio=4/3))
    scene.add_node(cam)
    scene.add_node(light)
    scene.add_node(node)
    cpose,mpose = np.eye(4),np.eye(4)
    cpose[:3,3] = [0,0,0.3]
    scene.set_pose(cam,cpose)
    #scene.set_pose(light,cpose)
    scene.set_pose(node,mpose)
    r = pyrender.OffscreenRenderer(viewport_width=640,viewport_height=480,point_size=1.0)

    #渲染前启用混合
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    color1,_ = r.render(scene)
    from matplotlib import pyplot as plt
    plt.imshow(color1)
    plt.axis('off')
    plt.show()
    return color1

face = render('../data/mesh/head_mesh.obj',3)
tooth = render('../data/mesh/teeth_double_layer.obj',1)
eroded_mesh = render('../data/mesh/teeth_double_layer_eroded.obj',1)
face_array = face.astype(float) / 255.0
tooth_array = tooth.astype(float) / 255.0
desired_alpha = 0.5
blend_result = desired_alpha * face_array + (1-desired_alpha) * tooth_array
blend = (blend_result*255).astype(np.uint8)
from matplotlib import pyplot as plt 
plt.imshow(blend)
plt.axis('off')
plt.show()

