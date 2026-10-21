
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from . import logging_setup # ロギング設定のインポートと実行
from .views.main_window import MainWindow
from .presenters.main_presenter import MainPresenter
from .models.main_model import MainModel
from .utils.path import AppPath

def main():
    # app を作成する前に属性を設定: これしないと、デフォルトダイアログになって、QMessageBox のアイコン表示がおかしくなる
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(AppPath.AppIcon)))

    # プラットフォームに依存しない一貫した外観を提供するスタイルを設定するなら、Fusionスタイルを使用
    # app.setStyle("Fusion") 

    view = MainWindow()
    model = MainModel()
    presenter = MainPresenter(view, model)
    view.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

