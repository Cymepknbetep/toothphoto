'''
主要职责为初始化相机、捕捉帧、检测棋盘格、求解位姿
'''

import cv2
import numpy as np
from typing import Tuple
from config import Config


class Camera:
    def __init__(self,config:Config):
        self.config = config
        self.cap = cv2.VideoCapture(config.camera_id)
        self._setup_camera()
        self.mtx,self.dist = self._init_calibration()
        self.obj_points = self._generate_chessboard_world()

    def _setup_camera(self) -> None:
        '''相机参数如分辨率和帧率'''
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 相当重要的格式设置
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.config.camera_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.config.camera_resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS,self.config.camera_fps)
        print(f'当前摄像头分辨率：{self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')    
        print(f'当前摄像头帧率：{self.cap.get(cv2.CAP_PROP_FPS)}')

    def _init_calibration(self) -> Tuple[np.ndarray,np.ndarray]:
        '''相机初始化内参和畸变参数'''   
        # 这里先进行简单假设
        focal_length = 1400 # pixel assume that cmos is around 5mm
        mtx = np.eye(3)
        mtx[0,0] = focal_length
        mtx[1,1] = focal_length
        mtx[:2,2] = [self.config.camera_resolution[0]/2,self.config.camera_resolution[1]/2]
        dist = np.zeros((5,1),dtype=np.float32) # assume that no distortion
        return mtx,dist
    
    def _generate_chessboard_world(self)->np.ndarray:
        objp = np.zeros((self.config.chessboard_size[0] * self.config.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.config.chessboard_size[0], 0:self.config.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.config.chessboard_square_size  # 乘以格子尺寸，转换为实际世界坐标
        # pick some points
        objp = objp[self.config.selected_indices]
        return objp   
    
    def capture_frame(self)->np.ndarray:
        '''捕捉一帧图像并且返回RGB格式'''
        ret,frame = self.cap.read()
        if not ret:
            raise ValueError("Frame capture failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # cv2是BGR需要转换一下

    def detect_chessboard(self,frame:np.ndarray) -> Tuple[bool, np.ndarray]:
        '''检测棋盘格角点'''
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        # TODO 超时控制
        ret, corners = cv2.findChessboardCorners(gray, self.config.chessboard_size, None)
        if ret:
            corners = cv2.cornerSubPix(image=gray,
                                        corners=corners,
                                        winSize=(11,11),
                                        zeroZone=(-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            selected_corners = corners[self.config.selected_indices]
            return ret,selected_corners
        else:
            return False,False

    def solve_pose(self,corners:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        '''solvePnP求解位姿，返回用于pyrender和位姿计算的两个矩阵'''
        # TODO better PnP
        ret,rvec,tvec = cv2.solvePnP(self.obj_points,corners,self.mtx,self.dist)
        if not ret:
            raise ValueError('PnP solve failed')
        R, _ = cv2.Rodrigues(rvec)
        # Pc = R @ Pw + tvec -> Pw = -R.T @ tvec
        t = -R.T @ tvec.flatten()
        pose_pyrender,camera_pose = np.eye(4),np.eye(4)
        # transform opencv->pyrender or pyrender-> opencv
        T_opencv2pyrender = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        pose_pyrender[:3,:3] = T_opencv2pyrender @ R.T @ T_opencv2pyrender  # We must add extra transform for later method
        # We could assosiate it with congruent transformation
        pose_pyrender[:3,3] = T_opencv2pyrender @ t
        camera_pose[:3,:3] = T_opencv2pyrender @ R
        camera_pose[:3,3] = T_opencv2pyrender @ t
        #print(pose_pyrender,camera_pose)

        return pose_pyrender,camera_pose

    def release(self)->None:
        '''释放相机资源'''
        self.cap.release()


















