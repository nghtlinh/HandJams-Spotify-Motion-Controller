import sys
from PyQt5.QtWidgets import QApplication
from ui.MainWindow import MainWindow

def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()