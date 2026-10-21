import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.models.rpd_model import AgilentDModel
from RAPID2_Agilent.views.data_stack import (
    ChromatogramSpectrumStackPresenter,
    ChromatogramSpectrumStackView,
)


DATA_PATH = Path("tests/data/dummy_RP_2min_neg.d")


def load_model():
    model = AgilentDModel()
    model.load(DATA_PATH)
    return model


def main():
    app = QApplication.instance() or QApplication(sys.argv)

    view = ChromatogramSpectrumStackView()
    presenter = ChromatogramSpectrumStackPresenter(view)
    presenter.add_model(load_model(), scan_index=5, mz_window=(150.0, 151.0))
    presenter.add_model(load_model(), scan_index=15, mz_window=(150.0, 151.0))
    view.presenter = presenter

    view.resize(1100, 700)
    view.setWindowTitle("Chromatogram / Spectrum Stack View")
    view.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
