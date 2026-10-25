
import argparse
from collections.abc import Sequence
import gc
from pathlib import Path
import sys
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from . import logging_setup # ロギング設定のインポートと実行
from .mcp import presence
from .views.fusion_style import AppFusionStyle
from .views.main_window import MainWindow
from .views.palette_fix import install_palette_fix
from .views.tooltip_position import install_tooltip_position
from .presenters.main_presenter import MainPresenter
from .models.main_model import MainModel
from .utils.path import AppPath

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAPID2 Agilent viewer.")
    parser.add_argument("folders", nargs="*", type=Path, help="Agilent .d folders to open at startup.")
    args = parser.parse_args(argv)
    missing_paths = [path for path in args.folders if not path.is_dir()]
    if missing_paths:
        parser.error("not found: " + ", ".join(str(path) for path in missing_paths))
    return args

def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    # held until exit, so that rapid2_launch sees this window even with AI access off
    window_presence = presence.hold()

    # app を作成する前に属性を設定: これしないと、デフォルトダイアログになって、QMessageBox のアイコン表示がおかしくなる
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(AppPath.AppIcon)))

    # Fusion ベースのスタイルとパレット補正 (ライト/ダークはOSの設定に追従)
    app.setStyle(AppFusionStyle())
    install_palette_fix(app)
    install_tooltip_position(app)

    view = MainWindow()
    model = MainModel()
    presenter = MainPresenter(view, model)
    app.aboutToQuit.connect(presenter.shutdown)
    if args.folders:
        presenter.open_d_folders_clicked([str(path.resolve()) for path in args.folders])
    presenter.apply_ai_assistant_setting()
    view.show()
    exit_code = app.exec()
    if window_presence is not None:
        window_presence.release()
    # インタプリタ終了時の PyQt cleanup_on_exit が解放済みラッパーを辿ってクラッシュするのを避けるため、
    # 生成と逆順に明示的に破棄してから終了する
    del presenter, model
    view.deleteLater()
    del view
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    del app
    gc.collect()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

