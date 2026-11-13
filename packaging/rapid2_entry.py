"""The entry script of the standalone build (RAPID2_Agilent.spec).

Every executable in the build is this script; each runs the program it is
named after (`RAPID2_Agilent.frozen`).
"""

import sys

from RAPID2_Agilent import frozen


if __name__ == "__main__":
    sys.exit(frozen.run())
