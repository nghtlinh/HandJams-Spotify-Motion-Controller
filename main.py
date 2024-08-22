import sys
from PyQt5.QtWidgets import *
from gui.main_window import main_window

def main():
    app = QApplication([])
    window = main_window()
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()