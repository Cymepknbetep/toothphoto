import sys
from main_window import MainWindow
from config import Config
from PyQt6.QtWidgets import QApplication

if __name__ == '__main__':
    config = Config(camera_test=True)
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec())