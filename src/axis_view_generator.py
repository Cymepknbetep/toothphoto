'''
预初始化fig/ax、更新箭头、缓存图像。分离出plt_init、create_axis等
'''

import matplotlib.pyplot as plt
import numpy as np
import hashlib
from typing import Tuple

from config import Config

class AxisViewGenerator:
    def __init__(self,config:Config):
        self.config = config
        # self.axis_cache = {}
        self._plt_init()

    def _plt_init(self) -> None:
        '''预初始化三个视图的fig/ax'''
        # 注意字体
        plt.rcParams['font.family'] = 'AR PL UKai CN'
        plt.rcParams['axes.unicode_minus'] = False
        self.fig_front,self.ax_front = self._create_pre_fig_ax('X','Y','正视图(X-Y)')
        self.fig_top,self.ax_top = self._create_pre_fig_ax('X','Z','俯视图(X-Z)')
        self.fig_side,self.ax_side = self._create_pre_fig_ax('Z','Y','侧视图(Z-Y)')
        #  初始化三个箭头对象
        self.arrow_front=self.arrow_top=self.arrow_side=None

    def _create_pre_fig_ax(self,xlabel:str,ylabel:str,title:str)-> Tuple[plt.figure,plt.Axes]:
        '''创建预绘制的fig/ax'''
        fig,ax = plt.subplots(figsize=(3,2),dpi=100) # 这里假设了300*200
        ax.set_facecolor = ('white')
        fig.patch.set_facecolor('white')
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        # 绘制不变的虚线十字轴
        ax.axhline(y=0,color='black',linestyle='--',linewidth=1, alpha=0.5)
        ax.axvline(x=0,color='black',linestyle='--',linewidth=1, alpha=0.5)
        # 设置不变的轴范围、标签、标题
        ax.set_xlim(-150,150)
        ax.set_ylim(-150,150)
        ax.set_xlabel(f'{xlabel}-                 {xlabel}+', color='black')
        ax.set_ylabel(f'{ylabel}-                 {ylabel}+', color='black')
        ax.set_title(title, color='black')
        fig.tight_layout()  # 优化布局
        return fig, ax        

    def create_axis(self,pose:np.ndarray)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        '''生成三个轴的视图图像，不使用缓存'''
        # pose_hash = hashlib.md5(pose.tobytes()).hexdigest()
        # if pose_hash in self.axis_cache:
        #     return self.axis_cache[pose_hash]
        # 计算投影、更新箭头、获取图像
        arrow_pose = pose.copy()
        z_axis = arrow_pose[:3, :3] @ np.array([0, 0, -1])  # 逆变换到世界系
        projection_front = z_axis[:2]  # x,y
        projection_top = np.array([z_axis[0], z_axis[2]])  # x,z
        projection_side = z_axis[1:][::-1]  # y,z (反转匹配原代码)
        # 获取更新图像
        img_front, self.arrow_front = self._update_arrow_and_get_img(self.fig_front, self.ax_front, projection_front, self.arrow_front)
        img_top, self.arrow_top = self._update_arrow_and_get_img(self.fig_top, self.ax_top, projection_top, self.arrow_top)
        img_side, self.arrow_side = self._update_arrow_and_get_img(self.fig_side, self.ax_side, projection_side, self.arrow_side)
        # # 储存缓存
        # self.axis_cache[pose_hash] = (img_front, img_top, img_side)
        return img_front, img_top, img_side
    
    def _update_arrow_and_get_img(self, fig: plt.Figure, ax: plt.Axes, projection: np.ndarray, arrow_ref) -> Tuple[np.ndarray, object]:
        '''更新箭头并从canvas获取图像'''
        # 移除旧箭头（如果存在）
        if arrow_ref is not None:
            arrow_ref.remove()
            
        # 缩放向量到 100 像素长度
        direction = projection * 200

        # 绘制新箭头，并更新引用
        new_arrow = ax.quiver(-direction[0]/2, -direction[1]/2, direction[0]/2, direction[1]/2, 
                                color='black', linewidth=4, scale=1, scale_units='xy', angles='xy')
            
        # 绘制到画布
        fig.canvas.draw()
            
        # 从 buffer 获取 ARGB 数据
        argb_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        argb_data = argb_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            
        # 转换为 RGB
        img_data = argb_data[:, :, :3]  # 跳过 A 通道
            
        return img_data, new_arrow  # 返回图像和新的箭头引用
















