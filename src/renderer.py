'''
初始化场景、渲染牙齿、面部、相机视图。分离出create_tooth_scene、create_camera_scene、render_tooth、render_camera等
'''

import pyrender
import trimesh
import numpy as np
from OpenGL import GL as gl

from abc import ABC, abstractmethod

from config import Config


class Renderer(ABC):
    @abstractmethod
    def render_tooth(self,pose:np.ndarray)->np.ndarray:
        pass

    @abstractmethod
    def render_camera(self,pose:np.ndarray) ->np.ndarray:
        pass

class PyrenderRenderer(Renderer):
    def __init__(self,config:Config):
        self.config = config
        self.renderer = pyrender.OffscreenRenderer(*config.render_size) # 解包参数  point_size代表渲染点云的点尺寸
        self._init_scenes()
        self.face_img = self._create_and_render_face_scene()
        #渲染前启用混合
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def _init_scenes(self)->None:
        '''初始化牙齿和相机的两个场景'''
        self.mesh_origin_trimesh = trimesh.load_mesh('../data/mesh/tooth_mesh.obj')
        self.scene_tooth = self._create_tooth_scene()
        self.scene_camera = self._create_camera_scene()

    def _create_tooth_scene(self)->pyrender.Scene:
        '''
        创建牙齿渲染的场景，包括mesh和材料、光照等等
        这里直接返回场景，同时需要使用属性存储光照和相机节点（其实光照在flatshading的设置下没有任何作用）
        '''
        # create scenes
        scene_tooth = pyrender.Scene(bg_color=[255,255,255])
        # init materials
        flat_shading_color_ratio = np.ones(1) # in order to get a darker img
        flat_shading_color_ratio[:3] = 0.3
        origin_color = [0.95,0.92,0.85,0.5] * flat_shading_color_ratio
        origin_eroded = [0.7,0.7,0.7,1] * flat_shading_color_ratio
        material_origin = pyrender.material.MetallicRoughnessMaterial(
            metallicFactor= 0,
            roughnessFactor= 0.3,
            baseColorFactor= origin_color,
            alphaMode='BLEND'
        )
        material_eroded = pyrender.material.MetallicRoughnessMaterial(
            metallicFactor= 0,
            roughnessFactor= 1,
            baseColorFactor = origin_eroded
        )
        # init meshes
        mesh_origin = pyrender.Mesh.from_trimesh(self.mesh_origin_trimesh,material=material_origin)
        mesh_eroded = pyrender.Mesh.from_trimesh(trimesh.load_mesh('../data/mesh/tooth_eroded_mesh.obj'),material=material_eroded)
        # init and add nodes
        nm_origin = pyrender.Node(mesh=mesh_origin)
        nm_eroded = pyrender.Node(mesh=mesh_eroded)
        self.nl_tooth = pyrender.Node(light=pyrender.PointLight(color=[1,1,1],intensity=1)) # note that when falt_shading is true, light is disabled
        self.nc_tooth = pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=np.pi/24,aspectRatio=self.config.render_size[0]/self.config.render_size[1]))
        scene_tooth.add_node(nm_origin)
        scene_tooth.add_node(nm_eroded)
        scene_tooth.add_node(self.nc_tooth)
        scene_tooth.add_node(self.nl_tooth)        
        # set poses
        scene_tooth.set_pose(nm_origin,self.config.cpose)
        scene_tooth.set_pose(nm_eroded,self.config.cpose)

        return scene_tooth


    def _create_camera_scene(self)->pyrender.Scene:
        '''
        创建相机渲染的场景，包括mesh和材料、光照等等
        这里直接返回场景，同时需要使用属性存储光照和相机节点、存储mesh
        '''
        # create scenes
        scene_camera = pyrender.Scene(bg_color=[255,255,255])
        # init materials
        material_cam = pyrender.material.MetallicRoughnessMaterial(
            metallicFactor=1,
            roughnessFactor=0.5,
            baseColorFactor=[1000,1000,0,1]
        )
        material_dot = pyrender.material.MetallicRoughnessMaterial(
            metallicFactor=0,
            roughnessFactor=0,
            baseColorFactor=[1000,0,0,1]
        )
        # init meshes
        mesh_camera = pyrender.Mesh.from_trimesh(trimesh.creation.cylinder(radius=0.01,height=self.config.arrow_length),
                                                     material=material_cam)
        mesh_dot = pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(subdivisions=2,radius=0.004),material=material_dot)
        # init and add nodes
        self.nm_camera = pyrender.Node(mesh=mesh_camera)
        self.nm_dot = pyrender.Node(mesh=mesh_dot)
        self.nl_camera = pyrender.Node(light=pyrender.DirectionalLight(color=[1,1,1],intensity=500))
        self.nc_camera = pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=np.pi/3,aspectRatio=self.config.render_size[0]/self.config.render_size[1]))
        scene_camera.add_node(self.nm_camera)
        scene_camera.add_node(self.nm_dot)
        scene_camera.add_node(self.nc_camera)
        scene_camera.add_node(self.nl_camera)
        # set poses
        scene_camera.set_pose(self.nc_camera,self.config.campose)

        return scene_camera

    def  _create_and_render_face_scene(self)->np.ndarray:
        '''预渲染面部和牙齿的混合图像'''
        # firstly create the face img
        # we need confirm the face position
        pose_face = self.config.cpose.copy()
        pose_face[:3,3] += [0,0.01,0]   # 调整，主要为了牙齿和面部图像的匹配
        # create scenes
        scene_face = pyrender.Scene(bg_color=[255,255,255])
        # init materials
        material_face = pyrender.material.MetallicRoughnessMaterial(
            metallicFactor= 0,
            roughnessFactor= 0.5,
            baseColorFactor= [0.82, 0.71, 0.59, 1]
        )
        # init meshes
        mesh_face = pyrender.Mesh.from_trimesh(trimesh.load_mesh('../data/mesh/head_mesh.obj'),material=material_face)
        # init and add nodes
        nm_face = pyrender.Node(mesh=mesh_face)
        #self.nl_face = pyrender.Node(light=pyrender.PointLight(color=[1,1,1],intensity=30))
        nl_face = pyrender.Node(light=pyrender.DirectionalLight(intensity=2))
        nc_face = pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=np.pi/3,aspectRatio=self.config.render_size[0]/self.config.render_size[1]))
        scene_face.add_node(nm_face)
        scene_face.add_node(nl_face)
        scene_face.add_node(nc_face)
        # set poses
        scene_face.set_pose(nm_face,pose_face)
        scene_face.set_pose(nc_face,self.config.campose)
        face,_ = self.renderer.render(scene_face)
        # then create the toothimg
        material_origin_camera = pyrender.material.MetallicRoughnessMaterial(
            metallicFactor= 0,
            roughnessFactor= 0.3,
            baseColorFactor= [0.95,0.92,0.85,1],
        )
        scene_tooth = pyrender.Scene(bg_color=[255,255,255])
        mesh_tooth = pyrender.Mesh.from_trimesh(self.mesh_origin_trimesh,material=material_origin_camera)
        nm_tooth = pyrender.Node(mesh=mesh_tooth)
        scene_tooth.add_node(nm_tooth)

        
        scene_tooth.add_node(nl_face)
        scene_tooth.add_node(nc_face)
        scene_tooth.set_pose(nm_tooth,self.config.cpose)
        scene_tooth.set_pose(nc_face,self.config.campose)
        tooth,_ = self.renderer.render(scene_tooth)

        # then render all of them
        face_img = (tooth*(1-self.config.mixed_alpha) + face*self.config.mixed_alpha).astype(np.uint8)
        return face_img

    def render_tooth(self,pose:np.ndarray) -> np.ndarray:
        '''渲染牙齿视图'''
        self.scene_tooth.set_pose(self.nl_tooth,pose)
        self.scene_tooth.set_pose(self.nc_tooth,pose)
        img,_ = self.renderer.render(self.scene_tooth,flags = pyrender.RenderFlags.FLAT)
        return img
    
    def render_camera(self, pose:np.ndarray)->np.ndarray:
        '''渲染相机视图，包括交互计算'''
        arrow_pose = pose.copy()
        # 传入的是相机to世界 rw = R @ rc
        z_axis = arrow_pose[:3,:3] @ np.array([0,0,-1])
        # 保证圆柱的中心在相机中心位置
        arrow_pose[:3,3] -= z_axis*self.config.arrow_length/2
        self.scene_camera.set_pose(self.nm_camera,pose)
        # now calculate interacts use trimesh
        locations,_,_ = self.mesh_origin_trimesh.ray.intersects_location(
            ray_origins = [pose[:3,3]-self.config.cpose[:3,3]],        # mainly because of the bias
            ray_directions = [z_axis],
            multiple_hits=False
        )
        # we define a far posision for when there's no interacts
        outpos = np.eye(4)
        outpos[:3,3] = [0,0,10]
        # 检测时候否获得交点
        if len(locations):
            locations += self.config.cpose[:3,3]
            pos = np.eye(4)
            pos [:3,3] = locations
            self.scene_camera.set_pose(self.nm_dot,pos)
        else: self.scene_camera.set_pose(self.nm_dot,outpos)
        # TODO acctually we could use depth info to blend
        camera,_ = self.renderer.render(self.scene_camera)
        temp = np.sum(camera, axis=2, dtype=np.uint16)  # 像素和，注意溢出
        mask = temp<=254*3
        mask = mask[:, :, np.newaxis]   # 保证存在新维度，事实上就是None的别名，为了保证可读性
        img = mask * camera + (~mask) * self.face_img
        img.astype(np.uint8)
        return img

    def cleanup(self)->None:
        '''释放渲染资源'''
        self.renderer.delete()















