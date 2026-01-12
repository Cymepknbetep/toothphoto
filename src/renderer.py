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
        self.mesh_origin_trimesh = trimesh.load_mesh('../data/mesh/teeth_double_layer.obj')
        #self.mesh_origin_trimesh = trimesh.load_mesh('../data/mesh/tooth_mesh.obj')

        self.scene_tooth = self._create_tooth_scene()
        self.scene_camera = self._create_camera_scene()
        self.scene_chessboard = self._create_chessboard_scene()

    def _create_chessboard_scene(self) -> pyrender.Scene:
        '''创建标定板渲染场景'''
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0]) # 背景透明（但在Offscreen中通常表现为黑色）
        
        # 计算标定板物理尺寸 (假设外围多出一个边框)
        width = (self.config.chessboard_size[0] + 1) * self.config.chessboard_square_size
        height = (self.config.chessboard_size[1] + 1) * self.config.chessboard_square_size
        
        # 创建一个薄板 (Box)，中心对齐
        # 注意：OpenCV 棋盘格定义在 Z=0 平面，这里创建一个稍微有一点厚度的板或者平面
        board_mesh = trimesh.creation.box(extents=[width, height, 0.001])
        
        # 将板的中心偏移，使得第一个角点(0,0,0)对应正确位置
        # trimesh.box 默认中心在原点，而 solvePnP 的原点是第一个内角点
        # 我们需要根据内角点数量将模型中心平移
        offset_x = (self.config.chessboard_size[0] - 1) * self.config.chessboard_square_size / 2
        offset_y = (self.config.chessboard_size[1] - 1) * self.config.chessboard_square_size / 2
        board_mesh.apply_translation([offset_x, -offset_y, 0])

        material = pyrender.material.MetallicRoughnessMaterial(
            baseColorFactor=[0.0, 1.0, 0.0, 0.5], # 绿色半透明，方便调试
            metallicFactor=0,
            roughnessFactor=1.0
        )
        
        mesh = pyrender.Mesh.from_trimesh(board_mesh, material=material)
        self.nm_chessboard = pyrender.Node(mesh=mesh)
        scene.add_node(self.nm_chessboard)

        # 添加相机节点（使用与牙齿渲染一致的相机参数）
        self.nc_chessboard = pyrender.Node(camera=pyrender.PerspectiveCamera(
            yfov=self.config.get_yfov(), 
            aspectRatio=self.config.camera_resolution[0]/self.config.camera_resolution[1]))
        scene.add_node(self.nc_chessboard)
        
        # 添加光照
        scene.add_node(pyrender.Node(light=pyrender.DirectionalLight(color=[1, 1, 1], intensity=2)))
        
        return scene
    
    def render_chessboard(self, pose: np.ndarray) -> np.ndarray:
        '''根据位姿渲染标定板平面'''
        # 标定板固定在世界原点，移动相机
        self.scene_chessboard.set_pose(self.nc_chessboard, pose)
        # 强制渲染分辨率与相机一致，以便叠加
        img, _ = self.renderer.render(self.scene_chessboard)
        return img

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
        mesh_eroded = pyrender.Mesh.from_trimesh(trimesh.load_mesh('../data/mesh/teeth_double_layer_eroded.obj'),material=material_eroded)
        # init and add nodes
        nm_origin = pyrender.Node(mesh=mesh_origin)
        nm_eroded = pyrender.Node(mesh=mesh_eroded)
        self.nl_tooth = pyrender.Node(light=pyrender.PointLight(color=[1,1,1],intensity=1)) # note that when falt_shading is true, light is disabled
        
        self.nc_tooth = pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=np.pi * self.config.teeth_fov,aspectRatio=self.config.render_size[0]/self.config.render_size[1]))
        scene_tooth.add_node(nm_origin)
        scene_tooth.add_node(nm_eroded)
        scene_tooth.add_node(self.nc_tooth)
        scene_tooth.add_node(self.nl_tooth)        
        # set poses
        scene_tooth.set_pose(nm_origin,self.config.get_teeth_matrix())
        scene_tooth.set_pose(nm_eroded,self.config.get_teeth_matrix())

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
        nc_face = pyrender.Node(camera=pyrender.PerspectiveCamera(
                                yfov=self.config.render_yfov, 
                                aspectRatio=self.config.render_size[0]/self.config.render_size[1]))
        scene_face.add_node(nm_face)
        scene_face.add_node(nl_face)
        scene_face.add_node(nc_face)
        # set poses
        scene_face.set_pose(nm_face,self.config.get_head_matrix())
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
        scene_tooth.set_pose(nm_tooth,self.config.get_teeth_matrix())
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
    
    # def render_camera(self, pose: np.ndarray) -> np.ndarray:
    #     '''渲染相机视图，采用 3x3 九宫格平行射线检测'''
    #     self.scene_camera.set_pose(self.nm_camera, pose)
        
    #     # 1. 参数设置
    #     radius = 0.002  # 探测半径（九宫格的外边距）
    #     x_axis = pose[:3, 0] 
    #     y_axis = pose[:3, 1] 
    #     z_axis = pose[:3, :3] @ np.array([0, 0, -1]) 

    #     # 射线中心起点（相对于原始 Mesh）
    #     cam_origin_local = pose[:3, 3] - self.config.teeth_trans

    #     # 2. 构造 3x3 九宫格平行射线起点
    #     # 偏移系数：[-1, 0, 1] 的组合
    #     offsets = [-1, 0, 1]
    #     origins = []
    #     for dx in offsets:
    #         for dy in offsets:
    #             # 计算每一根射线在相机平面的偏移位置
    #             origin_point = cam_origin_local + (x_axis * dx * radius) + (y_axis * dy * radius)
    #             origins.append(origin_point)
        
    #     directions = [z_axis] * len(origins) # 9 根射线全部保持平行

    #     # 3. 批量检测
    #     locations, index_ray, index_tri = self.mesh_origin_trimesh.ray.intersects_location(
    #         ray_origins=origins,
    #         ray_directions=directions,
    #         multiple_hits=False
    #     )

    #     # 4. 寻找到相机最近的交点
    #     outpos = np.eye(4)
    #     outpos[:3, 3] = [0, 0, 10]

    #     if len(locations) > 0:
    #         # 计算到相机起点的欧式距离
    #         distances = np.linalg.norm(locations - cam_origin_local, axis=1)
    #         closest_idx = np.argmin(distances)
            
    #         # 还原到世界坐标
    #         world_hit_point = locations[closest_idx] + self.config.teeth_trans
            
    #         pos_dot = np.eye(4)
    #         pos_dot[:3, 3] = world_hit_point
    #         self.scene_camera.set_pose(self.nm_dot, pos_dot)
    #     else:
    #         self.scene_camera.set_pose(self.nm_dot, outpos)

    #     # 5. 渲染与合成（保持不变）
    #     self.scene_camera.set_pose(self.nc_camera, self.config.campose)
    #     camera_rendering, _ = self.renderer.render(self.scene_camera)
        
    #     temp = np.sum(camera_rendering, axis=2, dtype=np.uint16)
    #     mask = (temp <= 254 * 3)[:, :, np.newaxis]
    #     img = (mask * camera_rendering + (~mask) * self.face_img).astype(np.uint8)
        
    #     return img

    def render_camera(self, pose: np.ndarray) -> np.ndarray:
        '''渲染相机视图，还原为单点射线检测（高性能版）'''
        # 1. 设置物理相机模型位姿
        self.scene_camera.set_pose(self.nm_camera, pose)
        
        # 2. 准备单根射线：中心方向与相对于原始 Mesh 的起点
        z_axis = pose[:3, :3] @ np.array([0, 0, -1]) 
        cam_origin_local = pose[:3, 3] - self.config.teeth_trans

        # 3. 执行单次检测
        locations, _, _ = self.mesh_origin_trimesh.ray.intersects_location(
            ray_origins=[cam_origin_local],
            ray_directions=[z_axis],
            multiple_hits=False
        )

        # 4. 处理结果
        outpos = np.eye(4)
        outpos[:3, 3] = [0, 0, 10]

        if len(locations) > 0:
            # 单点检测直接取第一个交点，加回平移量还原到世界系
            world_hit_point = locations[0] + self.config.teeth_trans
            
            pos_dot = np.eye(4)
            pos_dot[:3, 3] = world_hit_point
            self.scene_camera.set_pose(self.nm_dot, pos_dot)
        else:
            self.scene_camera.set_pose(self.nm_dot, outpos)

        # 5. 视角同步与渲染
        self.scene_camera.set_pose(self.nc_camera, self.config.campose)
        camera_rendering, _ = self.renderer.render(self.scene_camera)
        
        # 6. 图像合成
        temp = np.sum(camera_rendering, axis=2, dtype=np.uint16)
        mask = (temp <= 254 * 3)[:, :, np.newaxis]
        img = (mask * camera_rendering + (~mask) * self.face_img).astype(np.uint8)
        
        return img

    def cleanup(self)->None:
        '''释放渲染资源'''
        self.renderer.delete()















