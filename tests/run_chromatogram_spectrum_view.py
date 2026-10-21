import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.models.rpd_model import AgilentDModel
from RAPID2_Agilent.views.data_view import (
    ChromatogramSpectrumPresenter,
    ChromatogramSpectrumView,
)


DATA_PATH = Path("tests/data/dummy_RP_2min_neg.d")


def main():
    app = QApplication.instance() or QApplication(sys.argv)

    model = AgilentDModel()
    model.load(DATA_PATH)

    view = ChromatogramSpectrumView()
    presenter = ChromatogramSpectrumPresenter(view, model)
    presenter.initialize(scan_index=0, mz_window=(150.0, 151.0))
    view.presenter = presenter

    view.resize(1000, 500)
    view.setWindowTitle("Chromatogram / Spectrum View")
    view.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
