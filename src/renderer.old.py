import pyrender
import trimesh
import numpy as np
from OpenGL import GL as gl
from abc import ABC, abstractmethod
from config import Config

class Renderer(ABC):
    @abstractmethod
    def render_tooth(self, pose: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def render_camera(self, pose: np.ndarray) -> np.ndarray:
        pass

class PyrenderRenderer(Renderer):
    def __init__(self, config: Config):
        self.config = config
        self.renderer = pyrender.OffscreenRenderer(*config.render_size)
        
        # 初始化场景
        self._init_scenes()
        
        # 预渲染面部（注意：如果现场需要实时调人头位置，face_img 建议在循环中更新，目前先保留初始渲染）
        self.face_img = self._create_and_render_face_scene()
        
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def _init_scenes(self) -> None:
        # 加载 Mesh
        self.mesh_origin_trimesh = trimesh.load_mesh('../data/mesh/tooth_mesh.obj')
        self.mesh_eroded_trimesh = trimesh.load_mesh('../data/mesh/tooth_eroded_mesh.obj')
        self.mesh_head_trimesh = trimesh.load_mesh('../data/mesh/head_mesh.obj')
        
        # 创建场景
        self.scene_tooth = self._create_tooth_scene()
        self.scene_camera = self._create_camera_scene()

    def _create_tooth_scene(self) -> pyrender.Scene:
        scene = pyrender.Scene(bg_color=[255, 255, 255])
        
        # 恢复你之前的材质参数
        color_ratio = 0.3 # 让颜色深一点
        origin_color = [0.95, 0.92, 0.85, 0.5] # 半透明牙齿
        eroded_color = [0.7, 0.7, 0.7, 1.0]    # 灰色腐蚀层
        
        mat_origin = pyrender.material.MetallicRoughnessMaterial(
            metallicFactor=0, roughnessFactor=0.3, 
            baseColorFactor=[c * color_ratio if i < 3 else c for i, c in enumerate(origin_color)],
            alphaMode='BLEND'
        )
        mat_eroded = pyrender.material.MetallicRoughnessMaterial(
            metallicFactor=0, roughnessFactor=1.0, 
            baseColorFactor=[c * color_ratio if i < 3 else c for i, c in enumerate(eroded_color)]
        )

        # 存储 Node
        self.nm_origin = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(self.mesh_origin_trimesh, material=mat_origin))
        self.nm_eroded = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(self.mesh_eroded_trimesh, material=mat_eroded))
        
        # 相机与灯光
        self.cam_tooth_obj = pyrender.PerspectiveCamera(yfov=self.config.get_yfov(), aspectRatio=self.config.render_size[0]/self.config.render_size[1])
        self.nc_tooth = pyrender.Node(camera=self.cam_tooth_obj)
        self.nl_tooth = pyrender.Node(light=pyrender.PointLight(color=[1, 1, 1], intensity=1))

        # 必须先 add 后 set_pose
        scene.add_node(self.nm_origin)
        scene.add_node(self.nm_eroded)
        scene.add_node(self.nc_tooth)
        scene.add_node(self.nl_tooth)
        
        return scene

    def _create_camera_scene(self) -> pyrender.Scene:
        scene = pyrender.Scene(bg_color=[255, 255, 255])
        
        # 1. 虚拟棋盘格 (Debug用)
        cols, rows = self.config.chessboard_size
        s = self.config.chessboard_square_size
        width, height = (cols + 1) * s, (rows + 1) * s
        plane = trimesh.creation.box(extents=[width, height, 0.0001])
        plane.apply_translation([(width/2 - s), -(height/2 - s), 0])
        self.nm_debug_board = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(plane, 
            material=pyrender.material.MetallicRoughnessMaterial(baseColorFactor=[0, 1, 0, 0.3], alphaMode='BLEND')))
        
        # 2. 这里的场景也需要包含牙齿，以便在虚拟世界看到它们
        # 我们可以复用之前的 Mesh 重新创建 Node
        self.nm_origin_camv = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(self.mesh_origin_trimesh, 
            material=pyrender.material.MetallicRoughnessMaterial(baseColorFactor=[0.9, 0.9, 0.8, 1.0])))
        
        # 3. 相机指示器和红点
        self.nm_camera_visual = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(
            trimesh.creation.cylinder(radius=0.01, height=self.config.arrow_length), 
            material=pyrender.material.MetallicRoughnessMaterial(baseColorFactor=[1, 1, 0, 1])))
        self.nm_dot = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(radius=0.004), 
            material=pyrender.material.MetallicRoughnessMaterial(baseColorFactor=[1, 0, 0, 1])))
        
        self.cam_camera_obj = pyrender.PerspectiveCamera(yfov=self.config.get_yfov(), aspectRatio=self.config.render_size[0]/self.config.render_size[1])
        self.nc_camera = pyrender.Node(camera=self.cam_camera_obj)
        
        # Add nodes
        scene.add_node(self.nm_debug_board)
        scene.add_node(self.nm_origin_camv)
        scene.add_node(self.nm_camera_visual)
        scene.add_node(self.nm_dot)
        scene.add_node(self.nc_camera)
        
        return scene

    def update_intrinsics(self):
        yfov = self.config.get_yfov()
        self.cam_tooth_obj.yfov = yfov
        self.cam_camera_obj.yfov = yfov

    def render_tooth(self, camera_pose: np.ndarray) -> np.ndarray:
        # 获取最新的实时调节位姿
        teeth_pose = self.config.get_teeth_matrix()
        self.scene_tooth.set_pose(self.nm_origin, teeth_pose)
        self.scene_tooth.set_pose(self.nm_eroded, teeth_pose)
        
        self.scene_tooth.set_pose(self.nc_tooth, camera_pose)
        self.scene_tooth.set_pose(self.nl_tooth, camera_pose)
        
        self.update_intrinsics()
        img, _ = self.renderer.render(self.scene_tooth, flags=pyrender.RenderFlags.FLAT)
        return img

    def render_camera(self, camera_pose: np.ndarray) -> np.ndarray:
        # 1. 隐藏绿色平面，显示模型和指示器
        self.nm_debug_board.mesh.is_visible = False    # 隐藏绿幕
        self.nm_origin_camv.mesh.is_visible = True     # 显示牙齿
        self.nm_camera_visual.mesh.is_visible = True   # 显示黄色圆柱
        self.nm_dot.mesh.is_visible = True             # 显示红点
        
        # 2. 更新模型在虚拟世界的位置
        self.scene_camera.set_pose(self.nm_origin_camv, self.config.get_teeth_matrix())
        self.scene_camera.set_pose(self.nm_camera_visual, camera_pose)
        self.scene_camera.set_pose(self.nc_camera, camera_pose)
        
        # 3. 射线检测 (基于当前牙齿位姿)
        temp_mesh = self.mesh_origin_trimesh.copy()
        temp_mesh.apply_transform(self.config.get_teeth_matrix())
        z_axis = camera_pose[:3, :3] @ np.array([0, 0, -1])
        locations, _, _ = temp_mesh.ray.intersects_location(ray_origins=[camera_pose[:3, 3]], ray_directions=[z_axis], multiple_hits=False)
        
        if len(locations) > 0:
            dot_pose = np.eye(4)
            dot_pose[:3, 3] = locations[0]
            self.scene_camera.set_pose(self.nm_dot, dot_pose)
        else:
            self.scene_camera.set_pose(self.nm_dot, trimesh.transformations.translation_matrix([0,0,10]))

        self.update_intrinsics()
        camera_render, _ = self.renderer.render(self.scene_camera)
        
        # 混合面部背景
        temp = np.sum(camera_render, axis=2, dtype=np.uint16)
        mask = (temp <= 254 * 3)[:, :, np.newaxis]
        img = (mask * camera_render + (~mask) * self.face_img).astype(np.uint8)
        return img

    def render_chessboard_overlay(self, camera_pose: np.ndarray) -> np.ndarray:
        '''右下角校准视图'''
        self.update_intrinsics()
        self.scene_camera.set_pose(self.nc_camera, camera_pose)
        
        # 切换可见性：只看绿色平面
        self.nm_debug_board.mesh.is_visible = True
        self.nm_origin_camv.mesh.is_visible = False
        self.nm_camera_visual.mesh.is_visible = False
        self.nm_dot.mesh.is_visible = False
        
        color, _ = self.renderer.render(self.scene_camera)
        return color

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

    def cleanup(self) -> None:
        self.renderer.delete()