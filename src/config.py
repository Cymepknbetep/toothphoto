import numpy as np
import trimesh
import json
import os
from typing import Tuple

class Config:
    def __init__(self, camera_test: bool = True):
        # 配置文件保存路径
        self.config_path = "deploy_config.json"
        
        # --- 1. 相机基础参数 ---
        self.campose = np.eye(4)
        self.campose[2,3] = 0.5  # 默认相机位置 Z=0.5m
        self.camera_test = camera_test
        self.camera_id = 0 
        self.camera_resolution: Tuple[int, int] = (1920, 1080) # 现场需确认
        self.camera_fps: int = 30
        
        # 关键：动态内参 (fy)。开始默认 1400，cx, cy 默认取分辨率中心
        self.fy = 1400.0  
        
        # --- 2. 棋盘格物理参数 (必须测准) ---
        self.chessboard_size: Tuple[int, int] = (10, 7) # 内角点数量
        self.chessboard_square_size = 0.015            # 单格尺寸：1.5cm = 0.015m
        self.selected_indices = list(range(20, 50))    # PnP 使用的角点索引
        
        # --- 3. 人头对齐参数 (相对于棋盘格原点的位移和缩放) ---
        self.head_trans = [0.0, 0.0, 0.0]  # [x, y, z] 单位：米
        self.head_scale = 1.0              # 缩放系数
        
        # --- 4. 牙齿对齐参数 (相对于棋盘格原点) ---
        # 建议牙齿也相对世界原点，这样你可以独立调整它是否进入口腔
        self.teeth_trans = [0.0, 0.0, 0.0]
        self.teeth_scale = 1.0

        # --- 5. 渲染与 UI 设置 ---
        self.render_size: Tuple[int, int] = (640, 480) 
        self.mixed_alpha: float = 0.5   # 叠加显示时的透明度
        self.ui_fps = 60
        self.arrow_length = 0.1         # 虚拟坐标轴长度
        
        # 固定位置（由模型本身决定的初始偏置，如果没有则设为 eye）
        self.cpose = np.eye(4) 

        # 启动时自动加载上次保存的校准参数
        self.load_from_file()

    # --- 辅助方法：生成变换矩阵 ---

    def get_head_matrix(self) -> np.ndarray:
        """获取人头的 4x4 变换矩阵"""
        T = trimesh.transformations.translation_matrix(self.head_trans)
        S = trimesh.transformations.scale_matrix(self.head_scale)
        return T @ S

    def get_teeth_matrix(self) -> np.ndarray:
        """获取牙齿的 4x4 变换矩阵"""
        T = trimesh.transformations.translation_matrix(self.teeth_trans)
        S = trimesh.transformations.scale_matrix(self.teeth_scale)
        return T @ S

    def get_yfov(self) -> float:
        """根据当前的 fy 计算 pyrender 需要的垂直 FOV"""
        return 2.0 * np.arctan(self.camera_resolution[1] / (2.0 * self.fy))

    # --- 持久化存储逻辑 ---

    def save_to_file(self) -> None:
        """将当前所有校准参数保存到 JSON"""
        data = {
            "fy": self.fy,
            "head_trans": self.head_trans,
            "head_scale": self.head_scale,
            "teeth_trans": self.teeth_trans,
            "teeth_scale": self.teeth_scale
        }
        try:
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"已保存校准参数至: {self.config_path}")
        except Exception as e:
            print(f"保存配置失败: {e}")

    def load_from_file(self) -> None:
        """从 JSON 加载校准参数"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                self.fy = data.get("fy", self.fy)
                self.head_trans = data.get("head_trans", self.head_trans)
                self.head_scale = data.get("head_scale", self.head_scale)
                self.teeth_trans = data.get("teeth_trans", self.teeth_trans)
                self.teeth_scale = data.get("teeth_scale", self.teeth_scale)
                print(f"已从 {self.config_path} 加载参数")
            except Exception as e:
                print(f"加载配置失败，使用默认参数: {e}")
        else:
            print("未找到配置文件，使用初始化默认值")