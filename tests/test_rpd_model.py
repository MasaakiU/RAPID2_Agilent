# -*- coding: utf-8 -*-

# 実行
# python -m tests.test_main

import sys
import time
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from pathlib import Path
from RAPID2_Agilent.models.rpd_model import AgilentDModel
from unittest.mock import patch, mock_open

def test_model_load_logic():
    model = AgilentDModel()
    t0 = time.time()

    # select files
    model.load(Path("tests/data/MSMS_data/9_PRJ_0-1-2_PA34-1_1uM_n1.d"))
    # model.load(Path("tests/data/dummy_RP_2min_neg.d"))
    
    # ### spectrum
    for scan_idx, RT in enumerate(model.RT_list):
        mz_list, inten_list = model.extract_spectrum(scan_index=scan_idx)
        print(mz_list[:20])
        print(inten_list[:20])
        return mz_list, inten_list

    ### chromatogram
    RT_list, inten_list = model.extract_chromatogram(low_mz=150, high_mz=151)
    print(RT_list, inten_list)
    # quit()

    print(time.time() - t0)
    return RT_list, inten_list

if __name__ == "__main__":
    # データ取得
    mz_list, inten_list = test_model_load_logic()


    # 設定
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    # 簡単なアプリの作成
    app = QApplication(sys.argv)
    plt = pg.plot(mz_list, inten_list, symbol=None)
    plt.plotItem.vb.setMouseMode(pg.ViewBox.RectMode)
    plt.setWindowTitle('Simple Plot')
    plt.setLabel('bottom', 'm/z')
    plt.setLabel('left', 'Intensity')
    sys.exit(app.exec())
