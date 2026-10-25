"""Run RAPID2 Agilent with .d folders preloaded; defaults to a small test set.

    python run_app_with_test_data.py [folder.d ...]
"""

from pathlib import Path
import sys

from RAPID2_Agilent.main import main


TEST_DATA_DIR = Path(__file__).resolve().parent / "tests" / "data"
DEFAULT_D_FOLDERS = [
    TEST_DATA_DIR / "dummy_RP_2min_neg.d",
    TEST_DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d",
    # TEST_DATA_DIR / "dummy_RP_2min_neg.d",
    # TEST_DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d",
]


if __name__ == "__main__":
    main(sys.argv[1:] or [str(path) for path in DEFAULT_D_FOLDERS])
