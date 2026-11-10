import os

import numpy as np
from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.views.data_view import SpectrumWindow


def make_window(peaks: dict[float, float]):
    """A spectrum window of Gaussian profile peaks {m/z: height} on m/z 100-200."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = SpectrumWindow()
    window.resize(600, 300)
    window.show()
    app.processEvents()
    mz = np.round(np.arange(100.0, 200.0, 0.01), 2)
    intensity = np.zeros_like(mz)
    for center, height in peaks.items():
        intensity += height * np.exp(-0.5 * ((mz - center) / 0.02) ** 2)
    window.set_primary_plot_data(mz, intensity)
    return app, window


def label_texts(window) -> list[str]:
    return [text for _, text in window.peak_labels.labels()]


def assert_apart(window):
    rects = [rect for rect, _ in window.peak_labels.labels()]
    for index, rect in enumerate(rects):
        assert window.vb().rect().contains(rect)
        assert not any(rect.intersects(other) for other in rects[index + 1:])


def test_labels_peak_tops_clear_of_taller_lines_and_more_when_zoomed_in():
    app, window = make_window({120.0: 600.0, 120.3: 1000.0, 150.0: 800.0, 170.0: 10.0, 180.0: 50.0})

    # to the right of 120, the taller line of 120.3 leaves no room; 170, at 1% of the view's top, is not labelled
    assert label_texts(window) == ["120.3000", "150.0000", "180.0000"]
    assert_apart(window)

    window.plotItem.setXRange(119.5, 121.0)
    assert label_texts(window) == ["120.3000", "120.0000"]
    assert_apart(window)
    window.close()
    app.processEvents()


def test_labels_keep_clear_of_the_plots_top_right_label():
    app, window = make_window({195.0: 1000.0, 150.0: 500.0})
    assert label_texts(window) == ["195.0000", "150.0000"]

    window.set_top_right_label_html("12.3456 - 12.5678 min")
    assert label_texts(window) == ["150.0000"]
    window.close()
    app.processEvents()


def test_labels_cleared_with_the_spectrum():
    app, window = make_window({150.0: 1000.0})
    assert label_texts(window) == ["150.0000"]

    window.set_primary_plot_data(np.array([]), np.array([]))
    assert label_texts(window) == []
    window.close()
    app.processEvents()


def test_each_centroid_is_labelled_as_a_peak():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = SpectrumWindow()
    window.resize(600, 300)
    window.show()
    app.processEvents()
    # 150.5, lower than the centroids on either side, is a peak of its own
    mz, intensity = np.array([120.0, 150.0, 150.5, 151.0, 180.0]), np.array([800.0, 1000.0, 300.0, 600.0, 500.0])
    window.set_primary_plot_data(mz, intensity, spectra="centroid")
    assert window.peak_labels.apex_mz.tolist() == [150.0, 120.0, 151.0, 180.0, 150.5]
    window.set_primary_plot_data(mz, intensity)
    assert window.peak_labels.apex_mz.tolist() == [150.0, 151.0]
    window.close()
    app.processEvents()
