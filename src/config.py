from typing import Tuple
import numpy as np

class Config:
    def __init__(self,camera_test:bool=True):
        
        self._init_camera(camera_test)
        self._init_chessboard()
        self._init_renderer()
        self.render_size: Tuple[int,int] = (640,480) # pyrender渲染分辨率
        self.mixed_alpha: float = 0.5 # 设置人头透明度

        self.ui_fps = 60   
        # camera init
        
    
        

    def load_from_file(self, file_path:str) -> None:
        '''从yaml文件加载config'''
        pass

    def _init_camera(self,camera_test:bool=True)->None:
        # 是否测试相机显示
        self.camera_test = camera_test
        self.camera_id = 0 # 相机的文件标识符
        self.camera_resolution: Tuple[int, int] = (1920,1080) # 相机分辨率
        self.camera_fps: int = 30  # 相机帧率
        self._init_chessboard()

    def _init_chessboard(self)->None:
        self.chessboard_size: Tuple[int,int] = (10,7) # 棋盘格尺寸 # actually using width x height is more convenient, size hear means corners
        self.chessboard_square_size = 0.01
        self.selected_indices = [x for x in range(20,50)]   # now we use center 20 points for test
        self.chessboard_origin_position = (0,0)   # set a position!!! means x pixel * y pixel
        cx = (self.chessboard_size[0] - 1) * self.chessboard_square_size / 2
        cy = -(self.chessboard_size[1] - 1) * self.chessboard_square_size / 2
        self.chessboard_center = (cx,cy)# center in meters

    def _init_renderer(self)->None:
        self.arrow_length = 0.1
        self.cpose,self.campose = np.eye(4),np.eye(4)
        self.cpose[:3,3] = [self.chessboard_center[0],self.chessboard_center[1],0]
        self.campose[:3,3] = [self.chessboard_center[0],self.chessboard_center[1],0.5]



