import argparse
import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.models.main_model import MainModel
from RAPID2_Agilent.presenters.main_presenter import MainPresenter
from RAPID2_Agilent.utils.path import AppPath
from RAPID2_Agilent.views.main_window import MainWindow


DEFAULT_D_FOLDERS = [
    Path("tests/data/dummy_RP_2min_neg.d"),
    Path("tests/data/MSMS_data/9_PRJ_0-1-2_PA34-1_1uM_n1.d"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAPID2 Agilent with test .d folders preloaded.",
    )
    parser.add_argument(
        "folders",
        nargs="*",
        type=Path,
        help="Optional .d folders to preload. Defaults to a small test set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder_paths = args.folders or DEFAULT_D_FOLDERS
    folder_paths = [path.resolve() for path in folder_paths]

    missing_paths = [path for path in folder_paths if not path.exists()]
    if missing_paths:
        for path in missing_paths:
            print(f"Missing test data folder: {path}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(AppPath.AppIcon)))

    view = MainWindow()
    model = MainModel()
    presenter = MainPresenter(view, model)
    view.presenter = presenter

    presenter.open_d_folders_clicked([str(path) for path in folder_paths])
    view.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
