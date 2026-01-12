'''
管理队列、启动/停止线程、生成图像。组合以上类，不直接处理渲染或相机。
'''

import threading
import queue
import time
import numpy as np
import time # for debug
import cv2
from config import Config
from camera import Camera
from renderer import PyrenderRenderer
from axis_view_generator import AxisViewGenerator

class ImageGenerator:
    def __init__(self, config:Config):
        self.config = config
        self.camera = Camera(config)
                
        self.axis_generator = AxisViewGenerator(config)
        self.image_queues = [queue.Queue(maxsize=3) for _ in range(6)]  # 初始化缓冲队列
        self.running = False    # 用于线程
        self.renderer = None

    def start_generating(self) -> None:
        '''启动处理线程的接口'''
        self.running = True
        threading.Thread(target=self._generate_images,daemon=True).start()  # 守护线程，适合当作后台进程

    def stop_generating(self) -> None:
        '''重置标签并释放资源'''
        self.running = False
        self.camera.release()
        if self.renderer is not None:
            self.renderer.cleanup()
            self.renderer = None

    def _generate_images(self) -> None:
        '''主循环，捕捉帧、求解位姿、渲染、放入队列'''
        if self.renderer is None:
            self.renderer = PyrenderRenderer(self.config)
        while self.running:
            frame = self.camera.capture_frame()
            
            a = time.time()
            ret, corners = self.camera.detect_chessboard(frame)
            if self.config.camera_test:
                # 相机调试，放入队列 [5]
                # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #rgb_frame = frame[:,:,::-1]
                # 在中心添加红点
                frame[self.config.camera_resolution[1]//2-3:self.config.camera_resolution[1]//2+3, self.config.camera_resolution[0]//2-3:self.config.camera_resolution[0]//2+3, :] = [255,0,0]
            debug_frame = frame.copy()
            if ret:
                pose_pyrender, camera_pose = self.camera.solve_pose(corners)
                tooth_img = self.renderer.render_tooth(pose_pyrender)
                self._put_image(tooth_img,1)
                camera_img = self.renderer.render_camera(pose_pyrender)
                self._put_image(camera_img,0)
                img_front, img_top, img_side = self.axis_generator.create_axis(camera_pose)
                self._put_image(img_front,2)
                self._put_image(img_top,3)
                self._put_image(img_side,4)
                if self.config.camera_test:
                    board_overlay = self.renderer.render_chessboard(pose_pyrender)
                    board_overlay = cv2.resize(board_overlay, self.config.camera_resolution)
                    debug_frame = cv2.addWeighted(debug_frame, 0.7, board_overlay, 0.7, 0)
            self._put_image(debug_frame,5)
            print("Time per frame:",time.time()-a) # for debug
            

    def _put_image(self,img:np.ndarray,i:int) -> None:
        '''为多张图片的加入创建统一的接口'''
        try:
            self.image_queues[i].put(img,timeout=0.001) # without timeout, code will stop hear forever
        except queue.Full:
            pass








