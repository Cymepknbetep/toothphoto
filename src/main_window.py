'''
继承自QMainWindow，主要负责标签管理和定时器更新，从队列更新图像
'''

from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
import queue
from config import Config
from image_generator import ImageGenerator


class MainWindow(QMainWindow):
    def __init__(self,config:Config):
        super().__init__() # 调用父类的init方法
        self.config = config
        self.image_generator = ImageGenerator(self.config)
        self.image_labels = []  # 初始化标签列表
        self._setup_ui()    # init ui layout
        self.image_generator.start_generating()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_images_from_queue)
        self.timer.start(int(1000/self.config.ui_fps)) # 设置界面帧率


    def _setup_ui(self) -> None:
        '''设置布局和标签'''
        # 创建主布局 (水平布局 for image1 and image2)
        main_layout = QHBoxLayout()
        # 左侧布局：image1 和下面的 image3, image4, image5
        left_layout = QVBoxLayout()
        # 先添加image1
        image1_label = QLabel(self)
        image1_label.setFixedSize(640,480)
        self.image_labels.append(image1_label)
        left_layout.addWidget(image1_label)
        # 左侧布局的下部布局
        left_bottom_layout = QHBoxLayout()
        for i in range(3):
            label = QLabel(self)
            label.setFixedSize(300,200)
            self.image_labels.append(label)
            left_bottom_layout.addWidget(label)
        left_layout.addLayout(left_bottom_layout)
        # 添加左侧布局
        main_layout.addLayout(left_layout)
        # 创建image2的QLabel
        image2_label = QLabel(self)
        image2_label.setFixedSize(640,480)
        self.image_labels.insert(1,image2_label)
        # 考虑是否需要调试
        if not self.config.camera_test:
            main_layout.addWidget(image2_label)
        else: 
            # 调试模式下在右侧窗口展示当前相机画面
            camera_test_layout = QVBoxLayout()
            camera_view_label = QLabel(self)
            camera_view_label.setFixedSize(self.config.camera_resolution[0]//3,self.config.camera_resolution[1]//3) # 1920*1080
            self.image_labels.insert(5,camera_view_label)
            camera_test_layout.addWidget(image2_label)
            camera_test_layout.addWidget(camera_view_label)
            main_layout.addLayout(camera_test_layout)
        # 设置主窗口的中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        # 白色背景
        central_widget.setStyleSheet("background-color: #FFFFFF;")
        self.setCentralWidget(central_widget)

    def update_images_from_queue(self) -> None:
        '''从队列更新图像到QLabel'''
        for i in range(len(self.image_labels)):
            try: 
                frame = self.image_generator.image_queues[i].get_nowait()
                # to QPixmap
                height, width, channel = frame.shape
                bytes_per_line = 3*width
                qimage = QImage(frame.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
                # qimage = qimage.rgbSwapped() # BGR -> RGB
                pixmap = QPixmap.fromImage(qimage)
                if not pixmap.isNull():
                    self.image_labels[i].setPixmap(pixmap.scaled(self.image_labels[i].size(),Qt.AspectRatioMode.KeepAspectRatio))
            except queue.Empty:
                # 空队列则跳过读取
                pass
            
    def closeEvent(self, event):
        self.image_generator.stop_generating()
        #event.accept()
        super().closeEvent(event)








