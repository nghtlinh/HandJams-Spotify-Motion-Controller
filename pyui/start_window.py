# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'start_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(518, 425)
        MainWindow.setMinimumSize(QtCore.QSize(518, 425))
        MainWindow.setMaximumSize(QtCore.QSize(518, 425))
        font = QtGui.QFont()
        font.setKerning(False)
        MainWindow.setFont(font)
        MainWindow.setWindowOpacity(1.0)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("QWidget #main_layout{\n"
"    background-image: url(:/background/bgc.png);\n"
"    background-repeat: no-repeat;\n"
"    background-position: bottom;\n"
"    background-color: #2a292e;\n"
"    border: 3px solid #1db954;\n"
"    border-radius: 5px;\n"
"    padding: 10px;\n"
"}\n"
"QPushButton{\n"
"font-size: 18px;\n"
"font-weight: bold;\n"
"color: #1db954;\n"
"border: 2px solid #1db954;\n"
"border-radius: 14px;\n"
"letter-spacing: 1px;\n"
"}\n"
"QPushButton:hover {\n"
"color: #121212;\n"
"background-color:     #1db954;\n"
"border: 2px solid #121212;\n"
"}")
        self.main_layout = QtWidgets.QWidget(MainWindow)
        self.main_layout.setMinimumSize(QtCore.QSize(518, 425))
        self.main_layout.setMaximumSize(QtCore.QSize(518, 425))
        self.main_layout.setBaseSize(QtCore.QSize(518, 425))
        self.main_layout.setAcceptDrops(False)
        self.main_layout.setAutoFillBackground(False)
        self.main_layout.setStyleSheet("")
        self.main_layout.setObjectName("main_layout")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.main_layout)
        self.verticalLayout_6.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_61 = QtWidgets.QVBoxLayout()
        self.verticalLayout_61.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.verticalLayout_61.setObjectName("verticalLayout_61")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.horizontalLayout.setContentsMargins(-1, 0, 0, 30)
        self.horizontalLayout.setSpacing(1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.main_layout)
        self.label.setMinimumSize(QtCore.QSize(0, 100))
        self.label.setStyleSheet("QLabel {\n"
"color: #1db954;\n"
"font-size: 20px;\n"
"font-weight: bold;\n"
"}")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.verticalLayout_61.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.horizontalLayout_2.setContentsMargins(-1, 50, -1, -1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setContentsMargins(-1, -1, -1, 30)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.start_btn = QtWidgets.QPushButton(self.main_layout)
        self.start_btn.setMinimumSize(QtCore.QSize(150, 40))
        self.start_btn.setMaximumSize(QtCore.QSize(170, 40))
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.start_btn.setFont(font)
        self.start_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.start_btn.setStyleSheet("QPushButton{\n"
"font-size: 18px;\n"
"font-weight: bold;\n"
"color: #1db954;\n"
"border: 2px solid #1db954;\n"
"border-radius: 14px;\n"
"letter-spacing: 1px;\n"
"}\n"
"QPushButton:hover {\n"
"color: #121212;\n"
"background-color:     #1db954;\n"
"border: 2px solid #121212;\n"
"}")
        self.start_btn.setFlat(False)
        self.start_btn.setObjectName("start_btn")
        self.verticalLayout_5.addWidget(self.start_btn)
        self.login_btn = QtWidgets.QPushButton(self.main_layout)
        self.login_btn.setMinimumSize(QtCore.QSize(150, 40))
        self.login_btn.setMaximumSize(QtCore.QSize(156, 40))
        self.login_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.login_btn.setObjectName("login_btn")
        self.verticalLayout_5.addWidget(self.login_btn)
        self.verticalLayout_3.addLayout(self.verticalLayout_5)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        self.verticalLayout_61.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.verticalLayout_61)
        self.verticalLayout_6.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.main_layout)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Spotify gesture controller"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:15pt;\"> SPOTIFY  </span></p><p align=\"center\"><span style=\" font-size:15pt;\"> GESTURE CONTROLLER</span></p></body></html>"))
        self.start_btn.setText(_translate("MainWindow", "Start"))
        self.login_btn.setText(_translate("MainWindow", "Login"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
